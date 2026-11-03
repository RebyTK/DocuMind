from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model(model_name: str = MODEL_NAME):
    """Load and return a Sentence Transformer model."""
    return SentenceTransformer(model_name)

def embed_chunks(chunks: list[dict], model)-> list[dict]:
    """Generate embeddings while preserving chunk text and metadata."""
    if not chunks:
        return []
    
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            **chunk,
            "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
        })
    return embedded_chunks

