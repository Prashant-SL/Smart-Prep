from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts and returns all text from a PDF file.
    """
    text_content = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_content.append(text)
    return "\n".join(text_content)
