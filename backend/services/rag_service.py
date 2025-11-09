# In services.py
import logging
import os
import torch
import faiss
import numpy as np
from typing import Tuple, Dict
from sentence_transformers import SentenceTransformer

from utils.text_cleaner import chunk_text, extract_numbered_questions
from groq import Groq, APIError

# --- Logger ---
logger = logging.getLogger("app")

GROQ_CHAT_MODEL = "llama-3.1-8b-instant"



embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name, device = 'cuda' if torch.cuda.is_available() else 'cpu')

try:
    groq_client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )
    logger.info("Groq client initalised successfully!")
except Exception as error:
    logger.error(f"Error initalizing Groq: {error}")
    groq_client = None
    
def generate_questions_with_rag(resume: str, jd: str, role: str) -> tuple[list[str], list[str]]:
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
        return ["Could you tell me about your past experience?"], [] # Return empty list for suggestions
    
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
        return ["What are your key strengths for this role?"], [] # Return empty list for suggestions

    # --- NEW: Initialize separate lists and limits ---
    match_questions = []
    gap_questions = []
    all_suggestions = [] # Suggestions only come from gaps

    MAX_MATCH_QUESTIONS = 15
    MAX_GAP_QUESTIONS = 5
    MAX_SUGGESTIONS = 5
    
    L2_THRESHOLD = 1.0 
    
    # --- 3. NEW: Combined RAG Loop (Match, Gap, Suggestion) ---
    logger.info("--- Starting Combined RAG Loop ---")
    
    for i, jd_chunk in enumerate(jd_chunks):
        try:
            # Check if all our limits are already met
            if (len(match_questions) >= MAX_MATCH_QUESTIONS and
                len(gap_questions) >= MAX_GAP_QUESTIONS and
                len(all_suggestions) >= MAX_SUGGESTIONS):
                logger.info("All generation limits reached. Breaking loop early.")
                break

            query_embedding = embedding_model.encode([jd_chunk])
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            distances, indices = index.search(query_embedding, k=1)
            
            distance = distances[0][0] if (indices.size > 0 and indices[0][0] != -1) else 999
            
            # --- CASE 1: It's a MATCH ---
            if distance <= L2_THRESHOLD:
                # Only generate if we are under the match limit
                if len(match_questions) < MAX_MATCH_QUESTIONS:
                    resume_index = indices[0][0]
                    relevant_resume_chunk = resume_chunks[resume_index]
                    logger.info(f"[Loop {i+1}/{len(jd_chunks)}] MATCH found. (Count: {len(match_questions)}/{MAX_MATCH_QUESTIONS})")

                    messages = [
                        {'role': 'system', 'content': f"You are an expert interviewer for a '{role}' position. Respond *only* with a numbered list of questions."},
                        {'role': 'user', 'content': f"A job requirement is: \"{jd_chunk}\"\nThe candidate's resume states: \"{relevant_resume_chunk}\"\nGenerate 1-2 specific technical interview questions based *only* on the relationship between these two statements."}
                    ]
                    
                    try:
                        output = groq_client.chat.completions.create(
                            messages = messages,
                            model = GROQ_CHAT_MODEL,
                            temperature = 0.0,
                            max_tokens = 100
                        )
                        generated_text = output.choices[0].message.content
                        logger.info(f"[Match Loop] LLM Raw Output: {generated_text}")
                        
                        questions = extract_numbered_questions(generated_text, max_questions=2)
                        if questions:
                            logger.info(f"   -> Generated {len(questions)} match questions.")
                            match_questions.extend(questions)
                    except APIError as error:
                        logger.error(f"Groq API error in Match loop: {error}")
                        continue
            
            # --- CASE 2: It's a GAP ---
            else:
                # We do two things in a gap: generate a question AND a suggestion
                
                # 2a. Generate GAP question (if under limit)
                if len(gap_questions) < MAX_GAP_QUESTIONS:
                    logger.info(f"[Loop {i+1}/{len(jd_chunks)}] GAP found. (Q Count: {len(gap_questions)}/{MAX_GAP_QUESTIONS})")
                    messages = [
                        {"role": "system", "content": f"You are an expert interviewer for a '{role}' position. Respond *only* with a numbered list of questions."},
                        {"role": "user", "content": f"A job requirement is: \"{jd_chunk}\"\nThe candidate's resume does not seem to mention this. Generate 1 probing interview question to assess the candidate's experience with this."}
                    ]
                    try:
                        output = groq_client.chat.completions.create(
                            messages = messages, model = GROQ_CHAT_MODEL, temperature = 0.0, max_tokens = 100
                        )
                        generated_text = output.choices[0].message.content
                        questions = extract_numbered_questions(generated_text, max_questions=1)
                        if questions:
                            logger.info(f"   -> Generated {len(questions)} gap question.")
                            gap_questions.extend(questions)
                    except APIError as e:
                        logger.error(f"Groq API error in Gap Loop (Question): {e}")
                        # Don't "continue" yet, we might still be able to generate a suggestion
                
                # 2b. Generate SUGGESTION (if under limit)
                if len(all_suggestions) < MAX_SUGGESTIONS:
                    logger.info(f"[Loop {i+1}/{len(jd_chunks)}] GAP found. (Suggestion Count: {len(all_suggestions)}/{MAX_SUGGESTIONS})")
                    messages = [
                        {"role": "system", "content": "You are an AI Career Assistant. Respond *only* with a single suggestion sentence. Example: \"Consider adding a project or skill bullet point highlighting your experience with 'React Hooks' to better match the job description.\""},
                        {"role": "user", "content": f"A job requirement is: \"{jd_chunk}\"\nThe candidate's resume does not seem to mention this. Write a 1-sentence resume suggestion for the candidate, assuming they have this skill."}
                    ]
                    try:
                        output = groq_client.chat.completions.create(
                            messages=messages, model=GROQ_CHAT_MODEL, temperature=0.0, max_tokens=100
                        )
                        suggestion_text = output.choices[0].message.content.strip()
                        if suggestion_text and len(suggestion_text) > 20: # Basic filter
                            logger.info(f"   -> Generated 1 resume suggestion.")
                            all_suggestions.append(suggestion_text)
                    except APIError as e:
                        logger.error(f"Groq API error in Gap Loop (Suggestion): {e}")
                        # If this fails, just continue to the next loop item
                        continue
                        
        except Exception as e:
            logger.warning(f"Error during RAG loop: {e}")
            continue

    # --- 4. Combine and Deduplicate (This part is mostly the same) ---
    all_questions = match_questions[:MAX_MATCH_QUESTIONS] + gap_questions[:MAX_GAP_QUESTIONS]
    
    # 7. Deduplicate and return
    final_questions = []
    seen = set()
    for q in all_questions:
        if q not in seen:
            final_questions.append(q)
            seen.add(q)
    
    final_suggestions = []
    seen_s = set()
    for s in all_suggestions: # This list is already limited by MAX_SUGGESTIONS
        if s not in seen_s:
            final_suggestions.append(s)
            seen_s.add(s)
    
    logger.info(f"Generated {len(final_questions)} unique questions and {len(final_suggestions)} unique suggestions via RAG.")
    
    # Add generic questions if RAG produced too few
    if len(final_questions) < 5:
        logger.warning("RAG produced < 5 questions, adding generic ones.")
        final_questions.extend([
            f"Can you walk me through your experience as it relates to the '{role}' role?",
            "What project are you most proud of and why?",
        ])

    # The final slice here is a good safety net,
    # ensuring you never return more than 20 questions (15+5) or 5 suggestions.
    return final_questions[:20], final_suggestions[:5]
