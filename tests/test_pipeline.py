import json
import sys
import pytest
from pathlib import Path
from docx import Document
import main
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle



def test_docx_pipeline_generates_markdown_and_metadata(
    tmp_path,
    mock_postgres_ingestion,
):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    docx_path = documents_dir / "report.docx"

    document = Document()
    document.add_heading("Quarterly Revenue", level=1)
    document.add_paragraph("Revenue increased by 12%.")

    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Quarter"
    table.rows[0].cells[1].text = "Revenue"
    document.save(docx_path)

    result = main.ingest_document_to_postgres(
        docx_path,
        embedding_model=object(),
    )

    assert result["inserted"] is True

    persisted_document = mock_postgres_ingestion["document"]
    embedded_chunks = mock_postgres_ingestion["embedded_chunks"]

    assert persisted_document["file_type"] == "docx"
    assert persisted_document["filename"] == "report.docx"

    combined_text = "\n".join(
        chunk["text"]
        for chunk in embedded_chunks
    )

    assert "Quarterly Revenue" in combined_text
    assert "Revenue increased by 12%" in combined_text
    assert "Quarter" in combined_text
    assert "Revenue" in combined_text

    assert all(
        len(chunk["embedding"]) == 384
        for chunk in embedded_chunks
    )




def test_pdf_pipeline_preserves_tables(
    tmp_path,
    mock_postgres_ingestion,
):
    pdf_path = tmp_path / "report.pdf"

    data = [
        ["Quarter", "Revenue"],
        ["Q1", "$100,000"],
        ["Q2", "$112,000"],
    ]

    document = SimpleDocTemplate(str(pdf_path))

    table = Table(data)
    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )

    document.build([table])

    result = main.ingest_document_to_postgres(
        pdf_path,
        embedding_model=object(),
    )

    assert result["inserted"] is True

    persisted_document = mock_postgres_ingestion["document"]
    embedded_chunks = mock_postgres_ingestion["embedded_chunks"]

    assert persisted_document["file_type"] == "pdf"

    combined_text = "\n".join(
        chunk["text"]
        for chunk in embedded_chunks
    )

    assert "Quarter" in combined_text
    assert "Revenue" in combined_text
    assert "Q1" in combined_text
    assert "$100,000" in combined_text
    assert "Q2" in combined_text
    assert "$112,000" in combined_text

    # PDF chunks should preserve page metadata.
    assert all(
        chunk["metadata"]["page_number"] is not None
        for chunk in embedded_chunks
    )


@pytest.fixture
def mock_postgres_ingestion(monkeypatch):
    """Mock PostgreSQL persistence while preserving real extraction."""

    captured = {}

    monkeypatch.setattr(
        main,
        "calculate_file_hash",
        lambda path: "test-pipeline-hash",
    )

    monkeypatch.setattr(
        main,
        "get_document_by_hash",
        lambda file_hash: None,
    )

    monkeypatch.setattr(
        main,
        "generate_document_id",
        lambda: "doc_pipeline_test",
    )

    # Keep real document extraction and chunking.
    # Mock only embedding generation and database persistence.
    def fake_embed_chunks(chunks, model):
        return [
            {
                **chunk,
                "embedding": [1.0] + [0.0] * 383,
            }
            for chunk in chunks
        ]

    def fake_persist(document, embedded_chunks):
        captured["document"] = document
        captured["embedded_chunks"] = embedded_chunks

        return {
            "inserted": True,
            "document_id": document["document_id"],
            "filename": document["filename"],
            "total_chunks": len(embedded_chunks),
        }

    monkeypatch.setattr(
        main,
        "embed_chunks",
        fake_embed_chunks,
    )

    monkeypatch.setattr(
        main,
        "persist_document_with_chunks",
        fake_persist,
    )

    return captured

