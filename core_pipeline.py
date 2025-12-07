# core_pipeline.py
import os
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.runnables.retry import RunnableRetry

from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError

# Load API key
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM (best cost/quality for structured tasks in Nov 2025)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)

# Retry exceptions (use renamed Timeout)
RETRY_EXCEPTIONS = (RateLimitError, APIError, APITimeoutError, APIConnectionError)

# Helper to create robust chains
def create_chain(llm, prompt: ChatPromptTemplate, parser):
    chain = prompt | llm | parser
    return chain.with_retry(
        retry_if_exception_type=RETRY_EXCEPTIONS,  # Now fixed
        stop_after_attempt=4,  # 1 original + 3 retries
        wait_exponential_jitter=True  # Enables exponential backoff with jitter
    )

# ============================
# Pydantic Models (Structured Output)
# ============================

class SkillsToolsOutput(BaseModel):
    skills: List[str] = Field(..., description="List of required skills (unique, concise phrases)")
    tools: List[str] = Field(..., description="List of required tools, frameworks, software, technologies")

class CandidateInfoOutput(BaseModel):
    name: str = Field(..., description="Full name of the candidate")
    email: str = Field("N/A", description="Email address or N/A if not found")
    linkedin: str = Field("N/A", description="LinkedIn URL or N/A if not found")

class PresenceOutput(BaseModel):
    presence: Dict[str, str] = Field(
        ...,
        description="Dictionary: key = exact skill/tool name, value = 'Yes' or 'No' "
                    "(case-insensitive match, synonyms allowed only if very close)"
    )

class ReportOutput(BaseModel):
    report: str = Field(..., description="Concise but detailed screening report (200-400 words): strengths, shortcomings, overall fit")
    recommended: bool = Field(..., description="True if this is a strong candidate worth interviewing")
    reason: str = Field(..., description="Very short reason (1 sentence)")

# ============================
# Parsers
# ============================

skills_parser = PydanticOutputParser(pydantic_object=SkillsToolsOutput)
candidate_parser = PydanticOutputParser(pydantic_object=CandidateInfoOutput)
presence_parser = PydanticOutputParser(pydantic_object=PresenceOutput)
report_parser = PydanticOutputParser(pydantic_object=ReportOutput)

# ============================
# Chains (with retry + structured output)
# ============================

# 1. Skills & Tools from JD
skills_prompt = ChatPromptTemplate.from_template(
    """You are an expert HR analyst. Extract ONLY the key required skills and tools/technologies from the job description.
    - Skills = abilities/expertise (e.g., "Machine Learning", "Team Leadership")
    - Tools = software, frameworks, languages, platforms (e.g., "Python", "AWS", "Docker")
    No duplicates. Keep phrases short and exact.

    {format_instructions}

    Job Description:
    {job_description}"""
).partial(format_instructions=skills_parser.get_format_instructions())

skills_chain = create_chain(llm, skills_prompt, skills_parser)

# 2. Candidate basic info
candidate_prompt = ChatPromptTemplate.from_template(
    """Extract the candidate's full name, email, and LinkedIn URL from the CV.
    If not found → use "N/A".

    {format_instructions}

    CV Text:
    {cv_text}"""
).partial(format_instructions=candidate_parser.get_format_instructions())

candidate_chain = create_chain(llm, candidate_prompt, candidate_parser)

# 3. Skills/Tools presence check
presence_prompt = ChatPromptTemplate.from_template(
    """Check which of these exact items are mentioned in the CV (case-insensitive, close synonyms OK only if obvious).
    Items to check: {items_list}

    Output ONLY 'Yes' or 'No' for each.

    {format_instructions}

    CV Text:
    {cv_text}"""
).partial(format_instructions=presence_parser.get_format_instructions())

presence_chain = create_chain(llm, presence_prompt, presence_parser)

# 4. Final report + recommendation
report_prompt = ChatPromptTemplate.from_template(
    """You are a senior recruiter. Write a professional screening report for this candidate.

    Required items presence:
    {presence}

    Job Description:
    {job_description}

    CV Text:
    {cv_text}

    {format_instructions}"""
).partial(format_instructions=report_parser.get_format_instructions())

report_chain = create_chain(llm, report_prompt, report_parser)

# ============================
# Helper Functions
# ============================

def extract_text(file_path: str) -> str:
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Only PDF and DOCX files supported")
        
        docs = loader.load()
        text = "\n".join(page.page_content for page in docs)
        return text.replace("\x00", "").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read {os.path.basename(file_path)}: {str(e)}")

def create_dataframe(skills: List[str], tools: List[str]) -> pd.DataFrame:
    columns = ["name", "email", "linkedin"] + skills + tools + ["report", "recommendations"]
    return pd.DataFrame(columns=columns)

def add_candidate_to_df(
    df: pd.DataFrame,
    name: str,
    email: str,
    linkedin: str,
    presence: Dict[str, str],
    report: str,
    recommendation: str
) -> pd.DataFrame:
    
    row = {
        "name": name,
        "email": email,
        "linkedin": linkedin,
        "report": report,
        "recommendations": recommendation
    }
    
    # Fill skills/tools columns
    for col in df.columns:
        if col in ["name", "email", "linkedin", "report", "recommendations"]:
            continue
        row[col] = presence.get(col, "No")
    
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

# ============================
# Main Batch Processor
# ============================

def process_batch(
    cv_paths: List[str],
    job_description: str,
    job_title: str
) -> str:
    """
    Processes all CVs and returns the path to the generated Excel file.
    """
    # Step 1: Extract skills/tools once (same for all CVs)
    skills_tools: SkillsToolsOutput = skills_chain.invoke({"job_description": job_description})
    skills = skills_tools.skills
    tools = skills_tools.tools
    all_items = skills + tools
    
    df = create_dataframe(skills, tools)
    
    # Step 2: Process each CV
    for i, path in enumerate(cv_paths):
        try:
            cv_text = extract_text(path)
            
            # Extract info
            info: CandidateInfoOutput = candidate_chain.invoke({"cv_text": cv_text})
            
            # Check presence
            items_str = ", ".join(all_items)
            presence_result: PresenceOutput = presence_chain.invoke({
                "items_list": items_str,
                "cv_text": cv_text
            })
            presence_dict = presence_result.presence
            
            # Generate report
            report_result: ReportOutput = report_chain.invoke({
                "presence": str(presence_dict),
                "job_description": job_description,
                "cv_text": cv_text
            })
            
            recommendation_str = f"{'Yes' if report_result.recommended else 'No'} - {report_result.reason}"
            
            # Add to dataframe
            df = add_candidate_to_df(
                df=df,
                name=info.name.strip(),
                email=info.email.strip(),
                linkedin=info.linkedin.strip(),
                presence=presence_dict,
                report=report_result.report.strip(),
                recommendation=recommendation_str
            )
            
        except Exception as e:
            # Never let one bad CV crash the whole batch
            error_row = {
                "name": os.path.basename(path),
                "email": "ERROR",
                "linkedin": "PROCESSING FAILED",
                "report": f"Error: {str(e)}",
                "recommendations": "No - Processing error"
            }
            df = pd.concat([df, pd.DataFrame([error_row])], ignore_index=True)
    
    # Save Excel
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in job_title)
    excel_path = f"CV Screening for {safe_title}.xlsx"
    df.to_excel(excel_path, index=False)
    
    return excel_path