import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, logger
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import tempfile
import traceback
import re
import faiss
import numpy as np
# from models.request_models import ResumeTextRequest
from models.request_models import ResumeTextRequest
from utils.text_cleaner import clean_text, chunk_text
from utils.pdf_extractor import extract_text_from_pdf

app = FastAPI()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("app")

# Model configuration as per the system
quantization_model_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # instead of bfloat16 for better GPU compatibility
    llm_int8_enable_fp32_cpu_offload=False,  # disable CPU offload, you don’t need it
)

# Model setup & Configuration
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = quantization_model_config,
    dtype = torch.bfloat16,
    device_map = "auto"
)

llm_pipeline = pipeline("text-generation", model = model, tokenizer = tokenizer)
logger.info("Generation LLM loaded successfully.")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name, device = 'cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Embedding model loaded successfully.')

def extract_numbered_questions(text: str, max_questions: int = 20):
    """
    Extracts questions from text that are in a numbered list or lines that look like questions.
    Returns up to max_questions items.
    """
    # Normalize newlines and remove carriage returns
    text = text.replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines()]

    # 1) Prefer explicit numbered lines like "1. Question?" or "1) Question?"
    numbered_re = re.compile(r'^\s*(\d{1,2})[\.\)\-\]]\s*(.+)$')
    questions = []
    for ln in lines:
        m = numbered_re.match(ln)
        if m:
            q = m.group(2).strip()
            # filter out code blocks or instructions
            if q and not q.lower().startswith("you are"):
                questions.append(q)

    # 2) If none found, fallback: pick lines that end with "?" or look like questions
    if not questions:
        for ln in lines:
            if ln.endswith('?') and len(ln) > 10:
                questions.append(ln)
            elif ln.lower().startswith(("tell me", "explain", "how", "what", "why", "describe", "walk me")):
                questions.append(ln)
            if len(questions) >= max_questions:
                break

    # 3) Final cleaning: remove code-looking lines or placeholders, dedupe while preserving order
    final = []
    seen = set()
    for q in questions:
        if '`' in q or q.strip().startswith('def ') or q.strip().startswith('import '):
            continue
        norm = re.sub(r'\s+', ' ', q).strip()
        if norm and norm not in seen:
            final.append(norm)
            seen.add(norm)
        if len(final) >= max_questions:
            break

    return final[:max_questions]

def build_strict_prompt(resume_text: str, job_description: str, target_role: str):
    """
    Build a strict prompt that enforces output format.
    We include an explicit small example of the output format to reduce hallucinations.
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    prompt = f"""
        You are an AI Career Assistant.

        Analyze the following job description and resume for the role of a Frontend Developer.
        Based on them, generate **20 commonly asked technical interview questions** 
        that the candidate should prepare for.

        Respond **only** with a numbered list of questions — no explanations, no code, no extra text.

        Resume:
        {resume_text}

        Job Description:
        {job_description}
        """

    return prompt

@app.get("/health")
def health_check():
    logger.info("Health check endpoint was called.")
    return { "status": "ok" }

@app.post("/analyse-resume")
async def analyse_resume(payload: ResumeTextRequest):
    try:
        resume_text = payload.resume_text
        job_description = payload.job_description
        desired_role = payload.desired_role

        logger.info(f"Received analyze request for role: {desired_role} (resume len={len(resume_text)} JD len={len(job_description)})")

        # Build strict prompt
        prompt = build_strict_prompt(resume_text, job_description, desired_role)

        # Generation params (deterministic, conservative)
        gen_kwargs = {
            "max_new_tokens": 420,      # should be enough for 20 short questions
            "temperature": 0.0,         # deterministic
            "top_p": 1.0,
            "do_sample": False,         # deterministic sampling off
            "repetition_penalty": 1.1,
            "return_full_text": False,  # IMPORTANT: ask pipeline to return only generated text (no prompt echo)
        }

        # Attempt generation (primary)
        output = llm_pipeline(prompt, **gen_kwargs)
        generated_text = output[0].get("generated_text", "") if isinstance(output, list) else str(output)

        # Defensive: if pipeline still returns the prompt + generated, strip prompt if present
        if generated_text.strip().startswith(prompt.strip()[:200]):
            # remove prompt echo if any (best-effort)
            generated_text = generated_text.replace(prompt, "").strip()

        # Post-process to extract questions
        questions = extract_numbered_questions(generated_text, max_questions=20)

        # If the generation looks invalid (no questions), retry once with a shorter, more constrained prompt & different settings
        if len(questions) == 0:
            logger.warning("Primary generation returned no questions, retrying with stricter sampling.")
            retry_prompt = build_strict_prompt(resume_text, job_description, desired_role) + "\n(Reply ONLY with numbered questions.)\n"
            retry_kwargs = gen_kwargs.copy()
            retry_kwargs.update({"temperature": 0.0, "do_sample": False, "max_new_tokens": 400})
            retry_output = llm_pipeline(retry_prompt, **retry_kwargs)
            retry_text = retry_output[0].get("generated_text", "") if isinstance(retry_output, list) else str(retry_output)
            if retry_text.strip().startswith(retry_prompt.strip()[:200]):
                retry_text = retry_text.replace(retry_prompt, "").strip()
            questions = extract_numbered_questions(retry_text, max_questions=20)

        # Final fallback: try to heuristically extract question-like sentences
        if len(questions) == 0:
            logger.warning("Retry also failed — falling back to heuristic question generation.")
            # naive split by sentences and pick sentences that look like questions or start with common prompts
            sentences = re.split(r'(?<=[\.\?\!])\s+', clean_text(resume_text))
            fallback = [s for s in sentences if s.strip().endswith('?') or s.strip().lower().startswith(("how", "what", "why", "describe", "tell me"))]
            # if fallback is empty, produce generic questions based on skills in resume/JD
            if not fallback:
                fallback = [
                    "Tell me about a project where you used React.",
                    "How do you optimize a web application's performance?",
                    "Explain your experience with integrating REST APIs into frontend apps.",
                    "Describe how you ensure accessibility in your UI code.",
                    "How do you approach state management in complex React applications?"
                ]
            # ensure unique and limit to 20
            seen = set()
            final_fb = []
            for s in fallback:
                s_norm = s.strip()
                if s_norm not in seen:
                    final_fb.append(s_norm if s_norm.endswith('?') else s_norm + '?')
                    seen.add(s_norm)
                if len(final_fb) >= 20:
                    break
            questions = final_fb

        # Trim to 20 and return
        questions = questions[:20]

        return {
            "role": desired_role,
            "interview_questions": questions
        }

    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing the resume.")
    
@app.post("/upload-resume")
async def upload_resume(
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
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract and clean text
        extracted_text = extract_text_from_pdf(tmp_path)
        cleaned_text = clean_text(extracted_text)
        cleaned_jd = clean_text(job_description)
        
        # Call your new RAG function
        questions = await generate_questions_with_rag(
            cleaned_text, 
            cleaned_jd, 
            desired_role
        )

        return {
            "role": desired_role,
            "interview_questions": questions
        }
        # Prepare model prompt (LLM call to be added later)
        prompt = f"""
            You are an AI Career Assistant.
            Analyze the following job description and resume for the role '{desired_role}'.
            Generate 20 commonly asked interview questions for this profile based on the candidate's experince.
            Format output as a numbered list only.
            
            Resume: {cleaned_text[:2500]}
            Job Description: {cleaned_jd}
            """

        # Generation params (deterministic, conservative)
        gen_kwargs = {
            "max_new_tokens": 420,      # should be enough for 20 short questions
            "temperature": 0.0,         # deterministic
            "top_p": 1.0,
            "do_sample": False,         # deterministic sampling off
            "repetition_penalty": 1.1,
            "return_full_text": False,  # IMPORTANT: ask pipeline to return only generated text (no prompt echo)
        }

        # Attempt generation (primary)
        output = llm_pipeline(prompt, **gen_kwargs)
        generated_text = output[0].get("generated_text", "") if isinstance(output, list) else str(output)

        # Defensive: if pipeline still returns the prompt + generated, strip prompt if present
        if generated_text.strip().startswith(prompt.strip()[:200]):
            # remove prompt echo if any (best-effort)
            generated_text = generated_text.replace(prompt, "").strip()

        # Post-process to extract questions
        questions = extract_numbered_questions(generated_text, max_questions=20)

        # If the generation looks invalid (no questions), retry once with a shorter, more constrained prompt & different settings
        if len(questions) == 0:
            logger.warning("Primary generation returned no questions, retrying with stricter sampling.")
            retry_prompt = build_strict_prompt(resume_text, job_description, desired_role) + "\n(Reply ONLY with numbered questions.)\n"
            retry_kwargs = gen_kwargs.copy()
            retry_kwargs.update({"temperature": 0.0, "do_sample": False, "max_new_tokens": 400})
            retry_output = llm_pipeline(retry_prompt, **retry_kwargs)
            retry_text = retry_output[0].get("generated_text", "") if isinstance(retry_output, list) else str(retry_output)
            if retry_text.strip().startswith(retry_prompt.strip()[:200]):
                retry_text = retry_text.replace(retry_prompt, "").strip()
            questions = extract_numbered_questions(retry_text, max_questions=20)

        # Final fallback: try to heuristically extract question-like sentences
        if len(questions) == 0:
            logger.warning("Retry also failed — falling back to heuristic question generation.")
            # naive split by sentences and pick sentences that look like questions or start with common prompts
            sentences = re.split(r'(?<=[\.\?\!])\s+', clean_text(resume_text))
            fallback = [s for s in sentences if s.strip().endswith('?') or s.strip().lower().startswith(("how", "what", "why", "describe", "tell me"))]
            # if fallback is empty, produce generic questions based on skills in resume/JD
            if not fallback:
                fallback = [
                    "Tell me about a project where you used React.",
                    "How do you optimize a web application's performance?",
                    "Explain your experience with integrating REST APIs into frontend apps.",
                    "Describe how you ensure accessibility in your UI code.",
                    "How do you approach state management in complex React applications?"
                ]
            # ensure unique and limit to 20
            seen = set()
            final_fb = []
            for s in fallback:
                s_norm = s.strip()
                if s_norm not in seen:
                    final_fb.append(s_norm if s_norm.endswith('?') else s_norm + '?')
                    seen.add(s_norm)
                if len(final_fb) >= 20:
                    break
            questions = final_fb

        # Trim to 20 and return
        questions = questions[:20]

        return {
            "role": desired_role,
            "interview_questions": questions
        }

    except Exception as e:
        logger.exception("Error extracting PDF text.")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_questions_with_rag(resume: str, jd: str, role: str) -> list[str]:
    """
    Generates interview questions using a RAG approach.
    Wraps prompts in Gemma-2's required chat tokens.
    """
    logger.info("Starting RAG generation...")
    
    # 1. Text Chunking
    resume_chunks = chunk_text(resume)
    jd_chunks = chunk_text(jd)
    
    logger.info(f"Total Resume Chunks: {len(resume_chunks)}, Total JD Chunks: {len(jd_chunks)}")
    
    if not resume_chunks or not jd_chunks:
        logger.warning("Could not generate chunks for resume or JD. Aborting RAG.")
        return ["Could you tell me about your past experience?"] # Fallback
    
    # 2. Resume chunks embedding to vector store
    try:
        resume_embeddings = embedding_model.encode(resume_chunks)
        if resume_embeddings.dtype != np.float32:
             resume_embeddings = resume_embeddings.astype(np.float32)

        d = resume_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(resume_embeddings)
        logger.info(f"Created FAISS index with {len(resume_chunks)} resume chunks.")
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return ["What are your key strengths for this role?"] # Fallback

    all_questions = []
    
    rag_gen_kwargs = {
        "max_new_tokens": 100,
        "temperature": 0.0,
        "do_sample": False,
        "return_full_text": False,
    }
    
    # --- 3. LOOP 1: Find MATCHES ---
    logger.info("--- Starting RAG Match Loop ---")
    for i, jd_chunk in enumerate(jd_chunks):
        try:
            query_embedding = embedding_model.encode([jd_chunk])
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            distances, indices = index.search(query_embedding, k=1)
            
            if indices.size == 0 or indices[0][0] == -1:
                continue 
                
            resume_index = indices[0][0]
            relevant_resume_chunk = resume_chunks[resume_index]
            
            logger.info(f"[Match Loop {i+1}/{len(jd_chunks)}] JD Chunk matched Resume Chunk {resume_index}.")

            # --- MODIFIED PROMPT ---
            rag_prompt = f"""<start_of_turn>user
You are an expert interviewer for a '{role}' position.
A job requirement is: "{jd_chunk}"
The candidate's resume states: "{relevant_resume_chunk}"
Generate 1-2 specific technical interview questions based *only* on the relationship between these two statements. The questions should be appropriate for a candidate interviewing for the '{role}' position.
Respond *only* with a numbered list of questions.<end_of_turn>
<start_of_turn>model
"""
            
            output = llm_pipeline(rag_prompt, **rag_gen_kwargs)
            generated_text = output[0].get("generated_text", "")
            
            logger.info(f"[Match Loop] LLM Raw Output: {generated_text}")

            questions = extract_numbered_questions(generated_text, max_questions=2)
            if questions:
                logger.info(f"   -> Generated {len(questions)} questions for this pair.")
                all_questions.extend(questions)
            
        except Exception as e:
            logger.warning(f"Error during RAG match loop: {e}")
            continue
            
    # --- 4. LOOP 2: Find GAPS (Gap Analysis) ---
    logger.info("--- Starting RAG Gap Analysis Loop ---")
    
    L2_THRESHOLD = 1.0 
    
    for i, jd_chunk in enumerate(jd_chunks):
        try:
            query_embedding = embedding_model.encode([jd_chunk])
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            distances, indices = index.search(query_embedding, k=1)
            
            distance = distances[0][0] if (indices.size > 0 and indices[0][0] != -1) else 999
            
            if distance > L2_THRESHOLD:
                logger.info(f"[Gap Loop {i+1}/{len(jd_chunks)}] JD Chunk has no good match (distance: {distance:.2f}). Generating gap question.")
                
                # --- MODIFIED PROMPT ---
                gap_prompt = f"""<start_of_turn>user
You are an expert interviewer for a '{role}' position.
A job requirement is: "{jd_chunk}"
The candidate's resume does not seem to mention this specific skill or experience.
Generate 1 probing interview question to assess the candidate's experience with this requirement.
The question should be appropriate for a candidate interviewing for the '{role}' position.
Respond *only* with a numbered list of questions.<end_of_turn>
<start_of_turn>model
"""
                
                output = llm_pipeline(gap_prompt, **rag_gen_kwargs)
                generated_text = output[0].get("generated_text", "")
                
                logger.info(f"[Gap Loop] LLM Raw Output: {generated_text}")
                
                questions = extract_numbered_questions(generated_text, max_questions=1)
                
                if questions:
                    logger.info(f"   -> Generated {len(questions)} gap question.")
                    all_questions.extend(questions)

        except Exception as e:
            logger.warning(f"Error during RAG gap analysis loop: {e}")
            continue

    # 7. Deduplicate and return
    final_questions = []
    seen = set()
    for q in all_questions:
        if q not in seen:
            final_questions.append(q)
            seen.add(q)
    
    logger.info(f"Generated {len(final_questions)} unique questions via RAG (matches + gaps).")
    
    # Add generic questions if RAG produced too few
    if len(final_questions) < 5:
        logger.warning("RAG produced < 5 questions, adding generic ones.")
        final_questions.extend([
            f"Can you walk me through your experience as it relates to the '{role}' role?",
            "What project are you most proud of and why?",
        ])

    return final_questions[:20] # Return up to 20 questions
