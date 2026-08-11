"""
Tool: semantic search over a Pinecone index of previously-seen/cached
product data. Embeddings via OpenAI's text-embedding-3-small.

Requires PINECONE_API_KEY and OPENAI_API_KEY in .env. INDEX_NAME
defaults to "product-catalog" -- create this index in your Pinecone
project (dimension 1536, to match text-embedding-3-small) before use.
"""

import os
from pinecone import Pinecone
from openai import OpenAI

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = "product-catalog"
EMBEDDING_MODEL = "text-embedding-3-small"

_pc_client = None
_index = None
_openai_client = None


def _get_index():
    """Lazily creates the Pinecone client/index connection on first use."""
    global _pc_client, _index

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "pinecone_tools is not configured yet -- set PINECONE_API_KEY "
            "in .env before calling query_pinecone()."
        )

    if _pc_client is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc_client.Index(INDEX_NAME)

    return _index


def _get_openai_client() -> OpenAI:
    global _openai_client

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "pinecone_tools is not configured yet -- set OPENAI_API_KEY "
            "in .env before calling query_pinecone()."
        )

    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return _openai_client


def _embed_query(text: str) -> list[float]:
    """Embeds a single string using OpenAI's text-embedding-3-small."""
    client = _get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def query_pinecone(query: str, top_k: int = 5) -> list[dict]:
    """
    Embeds `query` and searches the Pinecone index for the closest matches.

    Returns a list of dicts shaped like:
        {"id": str, "score": float, "metadata": dict}
    """
    index = _get_index()
    vector = _embed_query(query)

    response = index.query(vector=vector, top_k=top_k, include_metadata=True)

    return [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
        }
        for match in response.get("matches", [])
    ]


def upsert_products(products: list[dict]) -> None:
    """
    Embeds and stores a batch of products in the index.

    Each product dict is expected to have at least "id" and "title"
    (title is what gets embedded); the full dict is stored as metadata
    so query_pinecone() results carry price/url/source directly.
    """
    index = _get_index()

    vectors = []
    for product in products:
        vector = _embed_query(product.get("title", ""))
        vectors.append({
            "id": product["id"],
            "values": vector,
            "metadata": product,
        })

    index.upsert(vectors=vectors)