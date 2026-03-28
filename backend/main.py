import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, logger
import tempfile
from utils.text_cleaner import clean_text, chunk_text, extract_numbered_questions
from utils.pdf_extractor import extract_text_from_pdf
from dotenv import load_dotenv
from services.rag_service import generate_questions_with_rag
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGINS")],
    allow_credentials=True,
    allow_methods=[os.environ.get("ALLOWED_METHODS")],
    allow_headers=[os.environ.get("ALLOWED_HEADERS")],
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail = "Only PDFs are allowed")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_contents = file.file.read() 
            tmp.write(file_contents)
            tmp_path = tmp.name

        # Extract and clean text
        extracted_text = extract_text_from_pdf(tmp_path)
        cleaned_text = clean_text(extracted_text)
        cleaned_jd = clean_text(job_description)
        
        logger.info(f"Received RAG upload request for role: {desired_role}")

        # Call your new RAG function
        questions, suggestions = generate_questions_with_rag(
            cleaned_text, 
            cleaned_jd, 
            desired_role
        )   

        return {
            "role": desired_role,
            "interview_questions": questions,
            "improvement_suggestions": suggestions
        }

    except Exception as e:
        logger.exception("Error extracting PDF text.")
        raise HTTPException(status_code=500, detail=str(e))
