import numpy as np

from processor.embeddings import embed_chunks

class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([
            [0.1, 0.2, 0.3] for _ in texts
        ])

def test_embed_chunks_preserves_metadata():
    chunks = [
        {
            "text": "Machine learning is interesting.",
            "metadata": {
                "source": "sample.pdf",
                "chunk_index": 0,
            },
        }
    ]
    model = FakeEmbeddingModel()

    result = embed_chunks(chunks, model)

    assert len(result) == 1
    assert result[0]["text"] == chunks[0]["text"]
    assert result[0]["metadata"] == chunks[0]["metadata"]
    assert result[0]["embedding"] == [0.1, 0.2, 0.3]

def test_embed_chunks_empty_input():

    result = embed_chunks([], FakeEmbeddingModel())

    assert result == []

def test_embed_chunks_generates_one_embedding_per_chunk():
    chunks = [
        {"text": "First chunk", "metadata": {"chunk_index": 0}},
        {"text": "Second chunk", "metadata": {"chunk_index": 1}},
        {"text": "Third chunk", "metadata": {"chunk_index": 2}},
    ]

    result = embed_chunks(chunks, FakeEmbeddingModel())
    assert len(result) == 3
    assert all("embedding" in chunk for chunk in result)
