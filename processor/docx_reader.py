from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

def extract_text_from_docx(file_path: Path) -> dict:
    """Extract paragraphs, lists, and tables in document order."""
    document = Document(file_path)
    elements = []

    body = document.element.body

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            from docx.text.paragraph import Paragraph

            paragraph = Paragraph(child, document)

            text = paragraph.text.strip()
            if not text:
                continue

            style_name = paragraph.style.name if paragraph.style else ""
            numbering = child.find(".//" + qn("w:numPr"))

            is_list_style = (
                "list" in style_name.lower()
                or "bullet" in style_name.lower()
                or "number" in style_name.lower()
            )

            if numbering is not None or is_list_style:
                element_type = "list_item"
            elif style_name.startswith("Heading"):
                element_type = "heading"
            else:
                element_type = "paragraph"

            elements.append({
                "type": element_type,
                "text": text,
                "style": style_name,
            })

        elif child.tag == qn("w:tbl"):
            from docx.table import Table

            table = Table(child, document)
            rows = []

            for row in table.rows:
                rows.append([
                    cell.text.strip()
                    for cell in row.cells
                ])

            elements.append({
                "type": "table",
                "rows": rows,
            })

    return {
        "filename": Path(file_path).name,
        "elements": elements,
    }
