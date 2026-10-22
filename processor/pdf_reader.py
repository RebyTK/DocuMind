from pathlib import Path
from pypdf import PdfReader

def extract_pdf_data(file_path:Path)->str:
    """Extract text page by page and report pages with no extractable text."""
    reader = PdfReader(file_path)

    pages =[]
    empty_text_pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": page_text,
        })

        if not page_text.strip():
            empty_text_pages.append(page_number)

    full_text = "\n".join(
        page["text"] for page in pages if page["text"].strip()
    )

    return {
        "filename": Path(file_path).name,
        "page_count": len(reader.pages),
        "text": full_text,
        "pages": pages,
        "pages_without_extractable_text": empty_text_pages,
    }


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract plain text from a PDF, preserving the existing interface."""
    return extract_pdf_data(file_path)["text"]