"""
retrieval.py

Shared RAG retrieval logic used by any agent that needs to look up
information from the Qdrant knowledge base (FAQ agent, billing agent).

Kept separate from individual agents so the embedding model and Qdrant
client are only loaded once and reused, instead of every agent loading
its own copy.
"""

import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "hospital_knowledge"

_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    return _client


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Given a natural language query, returns the top_k most relevant
    chunks from the knowledge base, concatenated into a single string
    ready to drop into a prompt.
    """
    model = _get_model()
    client = _get_client()

    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [f"[Source: {r.payload['source']}]\n{r.payload['text']}" for r in results]
    return "\n\n".join(chunks)
