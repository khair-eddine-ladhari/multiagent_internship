"""
main.py
-------
FastAPI entry point. Exposes POST /flow, called by node-client/apiClient.js.

Pipeline: main.py -> crew.py -> search_tools.py
"""

import logging
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

# Configure logging before anything else imports/uses a logger (e.g.
# search_tools.py), so INFO/WARNING messages from tool calls and the
# crew actually show up in the console instead of being silently
# dropped by the default root logger config.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import RetryError

from crew import run_product_search
import memory
import database

app = FastAPI(title="Product Assistant Server (CrewAI)")


class FlowRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    session_state: Optional[dict[str, Any]] = None


class FlowResponse(BaseModel):
    session_id: str
    answer: str
    session_state: dict[str, Any]


@app.post("/flow", response_model=FlowResponse)
def flow(request: FlowRequest) -> FlowResponse:
    session_id = request.session_id or str(uuid.uuid4())

    # Seed the in-memory session store with whatever state the client sent
    # back from the previous turn — the client is the durable copy, since
    # memory._sessions is process-local (see memory.py).
    incoming_state = request.session_state or {}
    memory.save_session(session_id, incoming_state)

    # Run the CrewAI crew: 3 store searches (Python, no LLM) -> comparator
    # agent. run_product_search already retries on RateLimitError internally
    # (see crew.py); if it still exhausts all retries, tenacity re-raises as
    # a RetryError. Catch that here so the client gets a clean 503 instead
    # of an unhandled 500 with a raw traceback.
    try:
        answer = run_product_search(request.query)
    except RetryError as e:
        logger.error("run_product_search exhausted retries for session=%s: %s", session_id, e)
        raise HTTPException(
            status_code=503,
            detail="The product search service is currently rate-limited. Please try again in a minute.",
        )
    except Exception as e:
        logger.exception("run_product_search failed unexpectedly for session=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Product search failed: {e}")

    # Update session memory with this turn's query/answer
    memory.record_turn(session_id, query=request.query, answer=answer)
    updated_state = memory.get_session(session_id)

    # Persist the turn to Mongo (best-effort; do not fail the request if this errors)
    try:
        database.save_turn(session_id=session_id, query=request.query, answer=answer)
    except Exception as e:
        logger.warning("database.save_turn failed (non-fatal) for session=%s: %s", session_id, e)

    return FlowResponse(
        session_id=session_id,
        answer=answer,
        session_state=updated_state,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)