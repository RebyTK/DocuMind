import json

from pathlib import Path

file_path = Path("output/sample_embeddings.json")

with file_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

# Chunks retrieved in your first evaluation
chunk_ids = [17, 1, 12, 20, 35, 69, 51, 13, 75]

print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    metadata = chunk.get("metadata", {})
    chunk_id = metadata.get("chunk_index")

    if chunk_id in chunk_ids:
        print("\n" + "=" * 80)
        print(f"CHUNK ID: {chunk_id}")
        print(f"SOURCE: {metadata.get('source', 'Unknown')}")
        print("-" * 80)
        print(chunk.get("text", ""))
