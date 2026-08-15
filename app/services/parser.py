import io
from pypdf import PdfReader

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    pdf_file = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def parse_txt(file_bytes: bytes) -> str:
    """Extract text from TXT file bytes, falling back to latin-1 if utf-8 fails."""
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1").strip()

def parse_document(filename: str, file_bytes: bytes) -> str:
    """Parse document based on its extension."""
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: .{ext}. Only PDF, TXT, and MD files are supported.")
