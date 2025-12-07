import os
import logging
from typing import List
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from core_pipeline import process_batch  # Import your function

load_dotenv()
app = FastAPI(title="CV Screening API", description="API for HR CV analysis")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def cleanup(cv_paths: List[str], excel_path: str):
    logger.info(f"Starting background cleanup for Excel: {excel_path}")
    for path in cv_paths:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(excel_path):
        os.remove(excel_path)
    logger.info("Cleanup complete")

@app.get("/")
async def root():
    return {
        "message": "CV Screening API is running."}

@app.get("/health")
async def health_check():
    return {
        "title": "CV Screening API",
        "python_version": os.sys.version,
        "framework": "FastAPI",
        "version": "1.0.0",
        "author": "Engr. Imtisal Ahmad Hashmi",
        "description": "API for HR CV analysis",
        "status": "healthy"}

@app.post("/process_cvs")
async def process_cvs(
    background_tasks: BackgroundTasks,  # MUST be first (no default)
    job_title: str = Form(...),
    job_description: str = Form(...),
    cv_files: List[UploadFile] = None
):
    if not cv_files or len(cv_files) == 0:
        raise HTTPException(status_code=400, detail="No CV files uploaded")

    cv_paths = []
    excel_path = None
    try:
        for file in cv_files:
            if not file.filename.lower().endswith((".pdf", ".docx")):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
            path = os.path.join(UPLOAD_DIR, file.filename)
            with open(path, "wb") as f:
                f.write(await file.read())
            cv_paths.append(path)

        logger.info(f"Processing {len(cv_paths)} CVs for job: {job_title}")
        excel_path = process_batch(cv_paths, job_description, job_title)
        logger.info(f"Excel generated at: {excel_path}")  # Confirm creation

        # Defer cleanup to AFTER response is fully sent
        background_tasks.add_task(cleanup, cv_paths, excel_path)

        return FileResponse(excel_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=os.path.basename(excel_path))

    except Exception as e:
        logger.error(f"Error processing: {str(e)}")
        # Immediate cleanup on error
        for path in cv_paths:
            if os.path.exists(path):
                os.remove(path)
        if excel_path and os.path.exists(excel_path):
            os.remove(excel_path)
        raise HTTPException(status_code=500, detail=str(e))