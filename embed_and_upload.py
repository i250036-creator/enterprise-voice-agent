"""
embed_and_upload.py

Re-embeds every .txt file in knowledge_base/ using fastembed and uploads
them to the Qdrant "hospital_knowledge" collection.

IMPORTANT: run this once after switching retrieval.py from
sentence-transformers to fastembed. Vectors from two different embedding
pipelines are NOT compatible with each other (different models produce
different vector spaces), so the collection needs to be recreated with
vectors generated the same way retrieval.py now generates its query vectors
— otherwise similarity search returns nonsense.

Usage:
    python embed_and_upload.py
"""

import os
import uuid
from pathlib import Path
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "hospital_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
CHUNK_SIZE = 700  # characters — rough proxy for the "500-800 token" target from the spec
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Simple sliding-window chunker. Not as smart as LangChain's
    RecursiveCharacterTextSplitter, but avoids pulling in another heavy
    dependency just for this — good enough for the small, clean knowledge
    base files in this project.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def main():
    if not KNOWLEDGE_BASE_DIR.exists():
        raise SystemExit(f"'{KNOWLEDGE_BASE_DIR}' folder not found. Run this from the project root.")

    txt_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in '{KNOWLEDGE_BASE_DIR}'.")

    print(f"Found {len(txt_files)} knowledge base file(s): {[f.name for f in txt_files]}")

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # Build all chunks across all files first, so we know the embedding
    # dimension before creating the collection.
    all_chunks = []  # list of (source_filename, chunk_text)
    for filepath in txt_files:
        content = filepath.read_text(encoding="utf-8")
        for chunk in chunk_text(content):
            all_chunks.append((filepath.name, chunk))

    print(f"Split into {len(all_chunks)} chunk(s). Embedding...")
    texts = [c[1] for c in all_chunks]
    vectors = list(model.embed(texts))
    vector_size = len(vectors[0])
    print(f"Embedding dimension: {vector_size}")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    print(f"Recreating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i].tolist(),
            payload={"source": all_chunks[i][0], "text": all_chunks[i][1]},
        )
        for i in range(len(all_chunks))
    ]

    print(f"Uploading {len(points)} point(s) to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("Done. Knowledge base re-embedded and uploaded with fastembed.")


if __name__ == "__main__":
    main()
