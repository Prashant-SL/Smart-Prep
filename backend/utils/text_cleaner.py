import re

def clean_text(text: str) -> str:
    """Remove extra whitespace, special chars, and normalize spacing."""
    # Replace weird unicode bullets or symbols with space
    text = re.sub(r'[•●▪️▶️]', ' ', text)
    # Remove multiple newlines or tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove page indicators (e.g., Page 2 of 3)
    text = re.sub(r'Page\s*\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)
    # Trim spaces
    text = text.strip()
    return text
    
def chunk_text(text: str) -> list[str]:
    """
    Splits text into meaningful chunks.
    Tries splitting by paragraphs, then lines, then sentences.
    Filters out chunks that are too short.
    """
    MIN_CHUNK_LEN = 50  # Chunks smaller than this (in chars) are probably not useful
    
    # 1. Try splitting by double newline (paragraphs)
    chunks = text.split("\n\n")
    cleaned_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_LEN]
    
    if len(cleaned_chunks) > 1:
        return cleaned_chunks

    # 2. If that fails, try splitting by single newline (lines)
    chunks = text.split("\n")
    cleaned_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_LEN]
    
    if len(cleaned_chunks) > 1:
        return cleaned_chunks

    # 3. If that fails, try splitting by sentences (final fallback)
    # This regex splits by '.', '!', '?' followed by a space or newline
    try:
        chunks = re.split(r'(?<=[\.\!\?])[\s\n]+', text)
        cleaned_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_LEN]
    except Exception as e:
        cleaned_chunks = [] # Force fallback

    if len(cleaned_chunks) > 1:
        return cleaned_chunks

    # 4. If all else fails, return the text as one chunk
    return [text.strip()] if len(text.strip()) > 0 else []

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