def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150,) -> list[str]:
    
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_text_with_metadata(text: str,source: str,chunk_size: int = 1000,overlap: int = 150,) -> list[dict]:
    
    """Split text into chunks and attach source metadata."""
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return [
        {
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_index": index,
            },
        }
        for index, chunk in enumerate(chunks)
    ]

