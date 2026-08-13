"""
memory.py
---------
Lightweight, process-local session memory.

Design note (see main.py): this store is NOT durable across server
restarts or multiple worker processes. The client (node-client) is
treated as the durable copy of session_state — it sends the previous
turn's state back on each request, and we seed our in-memory store
from that. Mongo (database.py) is the actual persistent log of turns,
used for history/analytics, not for serving state back to the client.
"""

from typing import Any
from datetime import datetime, timezone

# session_id -> session_state dict
_sessions: dict[str, dict[str, Any]] = {}


def save_session(session_id: str, state: dict[str, Any]) -> None:
    """Seed or overwrite the in-memory state for a session with client-provided state."""
    existing = _sessions.get(session_id, {})
    existing.update(state or {})
    _sessions[session_id] = existing


def get_session(session_id: str) -> dict[str, Any]:
    """Return the current in-memory state for a session (empty dict if unknown)."""
    return _sessions.get(session_id, {})


def record_turn(session_id: str, query: str, answer: str) -> None:
    """Append this turn's query/answer to the session's turn history."""
    state = _sessions.setdefault(session_id, {})
    turns = state.setdefault("turns", [])
    turns.append(
        {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    # Keep the in-memory history bounded; Mongo has the full record.
    if len(turns) > 20:
        state["turns"] = turns[-20:]


def clear_session(session_id: str) -> None:
    """Drop a session's in-memory state entirely."""
    _sessions.pop(session_id, None)