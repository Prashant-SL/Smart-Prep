import os
import logging
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from utils.text_cleaner import clean_text
from utils.pdf_extractor import extract_text_from_pdf
from services.rag_service import generate_questions_with_rag

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI()

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("app")

@app.get("/health")
def health_check():
    logger.info("Health check endpoint was called.")
    return { "status": "ok" }


@app.post("/upload-resume")
def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    desired_role: str = Form(...)
):
    """
    Accepts resume as a PDF file, job description, and desired role.
    Extracts text from the PDF, cleans it, and prepares it for LLM analysis.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        extracted_text = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    cleaned_text = clean_text(extracted_text)
    cleaned_jd = clean_text(job_description)

    logger.info(f"RAG request for role: {desired_role}")

    try:
        questions, suggestions = generate_questions_with_rag(
            cleaned_text, cleaned_jd, desired_role
        )
    except Exception as e:
        logger.exception("RAG generation failed.")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "role": desired_role,
        "interview_questions": questions,
        "improvement_suggestions": suggestions
    }