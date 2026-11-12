
from processor.chunker import (
    chunk_text_with_metadata,
    generate_document_id,
)


def test_generate_document_id():
    first_id = generate_document_id()
    second_id = generate_document_id()

    assert first_id.startswith("doc_")
    assert first_id != second_id


def test_chunks_have_unique_chunk_ids():
    document_id = "doc_test123"

    chunks = chunk_text_with_metadata(
        text="abcdefgh",
        source="sample.pdf",
        document_id=document_id,
        file_type="pdf",
        page_number=1,
        chunk_size=4,
        overlap=2,
    )

    assert len(chunks) == 3

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_metadata_preserves_document_information():
    chunks = chunk_text_with_metadata(
        text="ISO 27001 information",
        source="iso27001.pdf",
        document_id="doc_test123",
        file_type="pdf",
        page_number=12,
    )

    metadata = chunks[0]["metadata"]

    assert chunks[0]["chunk_id"] == "doc_test123_chunk_0000"
    assert metadata["document_id"] == "doc_test123"
    assert metadata["filename"] == "iso27001.pdf"
    assert metadata["file_type"] == "pdf"
    assert metadata["chunk_index"] == 0
    assert metadata["page_number"] == 12


def test_chunk_index_can_continue_across_pages():
    document_id = "doc_test123"

    page_one = chunk_text_with_metadata(
        text="abcdefgh",
        source="iso27001.pdf",
        document_id=document_id,
        file_type="pdf",
        page_number=1,
        chunk_size=4,
        overlap=0,
    )

    page_two = chunk_text_with_metadata(
        text="ijklmnop",
        source="iso27001.pdf",
        document_id=document_id,
        file_type="pdf",
        page_number=2,
        chunk_size=4,
        overlap=0,
        chunk_index_start=len(page_one),
    )

    all_chunks = page_one + page_two

    assert [c["metadata"]["chunk_index"] for c in all_chunks] == [
        0, 1, 2, 3
    ]

    assert len({
        c["chunk_id"] for c in all_chunks
    }) == 4

    assert all_chunks[0]["metadata"]["page_number"] == 1
    assert all_chunks[2]["metadata"]["page_number"] == 2


def test_docx_chunk_can_have_no_page_number():
    chunks = chunk_text_with_metadata(
        text="Quarterly revenue increased.",
        source="report.docx",
        document_id="doc_docx123",
        file_type="docx",
    )

    assert chunks[0]["metadata"]["page_number"] is None
