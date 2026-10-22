
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from processor.pdf_table_reader import extract_tables_from_pdf


def test_extract_tables_from_pdf(tmp_path: Path):
    pdf_path = tmp_path / "table.pdf"

    data = [
        ["Quarter", "Revenue"],
        ["Q1", "$100,000"],
        ["Q2", "$112,000"],
    ]

    document = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    table = Table(data)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )
    document.build([table])

    result = extract_tables_from_pdf(pdf_path)

    assert len(result) == 1
    assert result[0]["page"] == 1
    assert result[0]["table_number"] == 1
    assert result[0]["rows"][0] == ["Quarter", "Revenue"]
    assert result[0]["rows"][1] == ["Q1", "$100,000"]
    assert result[0]["rows"][2] == ["Q2", "$112,000"]
