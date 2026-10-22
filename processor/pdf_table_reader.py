from pathlib import Path
import pdfplumber


def extract_tables_from_pdf(file_path: Path) -> list[dict]:
    """Extract detected PDF tables, preserving page numbers."""
    extracted_tables = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_number, table in enumerate(
                page.extract_tables(),
                start=1,
            ):
                if not table:
                    continue

                rows = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in table
                ]

                # Skip tables that contain no usable cell values.
                if not any(cell for row in rows for cell in row):
                    continue

                extracted_tables.append({
                    "page": page_number,
                    "table_number": table_number,
                    "rows": rows,
                })

    return extracted_tables

