from pathlib import Path
from reportlab.pdfgen import canvas
from processor.pdf_reader import extract_text_from_pdf, extract_pdf_data


def test_extract_text_from_pdf(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "DocuMind PDF extraction works")
    pdf.save()

    result = extract_text_from_pdf(pdf_path)

    assert "DocuMind PDF extraction works" in result


def test_extract_text_from_multiple_pages(tmp_path: Path):
    pdf_path = tmp_path / "multiple_pages.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "First page")
    pdf.showPage()

    pdf.drawString(72, 720, "Second page")
    pdf.save()

    result = extract_text_from_pdf(pdf_path)

    assert "First page" in result
    assert "Second page" in result



def test_extract_pdf_data_preserves_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "pages.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "Page one content")
    pdf.showPage()

    pdf.drawString(72, 720, "Page two content")
    pdf.save()

    result = extract_pdf_data(pdf_path)

    assert result["page_count"] == 2
    assert result["pages"][0]["page"] == 1
    assert result["pages"][1]["page"] == 2
    assert "Page one content" in result["pages"][0]["text"]
    assert "Page two content" in result["pages"][1]["text"]
    assert result["pages_without_extractable_text"] == []
