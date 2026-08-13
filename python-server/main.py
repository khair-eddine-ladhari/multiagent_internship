"""
main.py
-------
FastAPI entry point. Exposes POST /flow, called by node-client/apiClient.js.

Old pipeline:   main.py -> flow.py -> agent.py -> search_tools.py (stubs)
New pipeline:   main.py -> crew.py  -> search_tools.py (real tools)

flow.py / agent.py are no longer imported anywhere in this file.
"""

import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

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

    # Run the CrewAI crew: 3 store search agents -> comparator/reporter agent
    answer = run_product_search(request.query)

    # Update session memory with this turn's query/answer
    memory.record_turn(session_id, query=request.query, answer=answer)
    updated_state = memory.get_session(session_id)

    # Persist the turn to Mongo (best-effort; do not fail the request if this errors)
    try:
        database.save_turn(session_id=session_id, query=request.query, answer=answer)
    except Exception as e:
        print(f"[main.py] database.save_turn failed (non-fatal): {e}")

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