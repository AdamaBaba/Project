from pathlib import Path


def read_document(path):
    """
    Reads TXT, DOCX and PDF essay files.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".txt":
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    if suffix == ".docx":
        from docx import Document

        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    raise ValueError("Unsupported file type. Please upload TXT, DOCX or PDF.")
