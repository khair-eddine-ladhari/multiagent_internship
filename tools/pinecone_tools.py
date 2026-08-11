"""
Tool: semantic search over a Pinecone index.

What this index actually stores (a product catalog, cached past
searches, reviews, etc.) isn't decided yet -- INDEX_NAME and the
embedding model are left as placeholders. The shape of query_pinecone()
is written generically: embed the query, search the index, return
matches. Fill in EMBEDDING_MODEL and INDEX_NAME once that's decided.
"""

import os
from pinecone import Pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = ""  # e.g. "product-catalog"
EMBEDDING_MODEL = ""  # e.g. "text-embedding-3-small" (OpenAI) or a Pinecone-hosted model

_pc_client = None
_index = None


def _get_index():
    """Lazily creates the Pinecone client/index connection on first use."""
    global _pc_client, _index

    if not PINECONE_API_KEY or not INDEX_NAME:
        raise RuntimeError(
            "pinecone_tools is not configured yet -- set PINECONE_API_KEY "
            "and INDEX_NAME before calling query_pinecone()."
        )

    if _pc_client is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc_client.Index(INDEX_NAME)

    return _index


def _embed_query(query: str) -> list[float]:
    """
    Turns a text query into a vector for similarity search.

    TODO: wire up the real embedding call once EMBEDDING_MODEL is
    decided (e.g. an OpenAI embeddings call, or Pinecone's hosted
    inference API).
    """
    raise NotImplementedError(
        "Set EMBEDDING_MODEL and implement _embed_query() before use."
    )


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

    Each product dict is expected to have at least an "id" and a text
    field to embed (e.g. "title" or "description") -- exact shape TBD
    once the catalog source is decided.
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