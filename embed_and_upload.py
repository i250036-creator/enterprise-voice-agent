"""
embed_and_upload.py

Reads all .txt files from knowledge_base/, splits them into chunks,
generates embeddings using a HuggingFace sentence-transformer model,
and uploads them to a Qdrant collection called 'hospital_knowledge'.

Run this once after adding/updating any file in knowledge_base/.
"""

import os
import glob
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ---- Load environment variables from .env ----
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hospital_knowledge"
KNOWLEDGE_BASE_DIR = "knowledge_base"

# ---- Step 1: Load the embedding model ----
# all-MiniLM-L6-v2 is small, fast, and free — good enough for FAQ-style retrieval.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = model.get_sentence_embedding_dimension()  # 384 for this model

# ---- Step 2: Connect to Qdrant ----
print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Recreate the collection fresh each time this script runs.
# This keeps things simple for development — in production you'd want incremental updates.
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print(f"Collection '{COLLECTION_NAME}' created.")

# ---- Step 3: Load and chunk documents ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # roughly 500-800 tokens as planned in Phase 2
    chunk_overlap=80,     # slight overlap so context isn't cut mid-sentence
    separators=["\n\n", "\n", ". ", " "],
)

all_chunks = []
all_metadata = []

txt_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))
print(f"Found {len(txt_files)} document(s) in {KNOWLEDGE_BASE_DIR}/")

for filepath in txt_files:
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)
    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadata.append({"source": filename, "text": chunk})

print(f"Total chunks created: {len(all_chunks)}")

# ---- Step 4: Generate embeddings ----
print("Generating embeddings...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# ---- Step 5: Upload to Qdrant ----
print("Uploading to Qdrant...")
points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload=all_metadata[idx],
    )
    for idx, embedding in enumerate(embeddings)
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Done. {len(points)} chunks uploaded to '{COLLECTION_NAME}'.")

# ---- Step 6: Quick test query ----
print("\n--- Running a test query ---")
test_query = "What are the clinic's emergency department hours?"
test_embedding = model.encode(test_query).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=test_embedding,
    limit=3,
).points

for i, result in enumerate(results, 1):
    print(f"\nResult {i} (score: {result.score:.3f}, source: {result.payload['source']})")
    print(result.payload["text"][:200] + "...")
