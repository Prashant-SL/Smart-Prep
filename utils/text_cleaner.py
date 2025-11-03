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