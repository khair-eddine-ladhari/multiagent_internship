"""
database.py
------------
MongoDB persistence layer for turn history.

This is the durable, cross-process log of every query/answer pair —
unlike memory.py's in-memory _sessions dict, this survives restarts
and is shared across worker processes.

main.py calls save_turn() inside a try/except and treats failures as
non-fatal (a DB outage shouldn't break the /flow response), so this
module intentionally does NOT swallow its own exceptions — it raises
and lets the caller decide.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

_client: Optional[MongoClient] = None
_collection: Optional[Collection] = None


def _get_collection() -> Collection:
    """Lazily create and cache the Mongo client/collection."""
    global _client, _collection

    if _collection is not None:
        return _collection

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.environ.get("MONGO_DB_NAME", "product_assistant")
    collection_name = os.environ.get("MONGO_COLLECTION_NAME", "turns")

    _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    _collection = _client[db_name][collection_name]
    return _collection


def save_turn(session_id: str, query: str, answer: str) -> None:
    """Append a query/answer turn to this session's document in Mongo."""
    collection = _get_collection()
    collection.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "turns": {
                    "query": query,
                    "answer": answer,
                    "timestamp": datetime.now(timezone.utc),
                }
            }
        },
        upsert=True,
    )


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    """Fetch the most recent turns for a session, oldest first."""
    collection = _get_collection()
    doc = collection.find_one({"session_id": session_id})
    if not doc:
        return []
    turns = doc.get("turns", [])
    return turns[-limit:]