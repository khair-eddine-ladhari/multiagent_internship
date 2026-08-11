"""
FastAPI entry point for the product search pipeline.

This file does NOT contain any pipeline/agent logic. It just:
  1. Imports the already-compiled LangGraph graph (app_graph)
  2. Validates incoming requests
  3. Runs the graph synchronously
  4. Shapes the graph's output into an HTTP response

Run with:
    uvicorn api.main:app --reload

Then:
    POST http://localhost:8000/search
    { "query": "gaming laptop under $1000" }
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph.build_graph import app_graph


app = FastAPI(
    title="Product Search Pipeline API",
    description="Query understanding -> multi-source search -> normalization -> ranking -> report",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language product search query")


class SearchResponse(BaseModel):
    query: str
    report: str
    ranked_products: list[dict]


class ClarificationResponse(BaseModel):
    query: str
    needs_clarification: bool = True
    clarification_question: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Simple liveness check for monitoring/deployment."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        result = app_graph.invoke({"user_query": query})
    except Exception as exc:
        # Don't leak internals; log server-side in a real deployment.
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    # If query understanding flagged the query as too vague, surface that
    # instead of returning an empty/fake report.
    if result.get("needs_clarification"):
        raise HTTPException(
            status_code=422,
            detail={
                "query": query,
                "needs_clarification": True,
                "clarification_question": result.get(
                    "clarification_question", "Could you clarify your query?"
                ),
            },
        )

    return SearchResponse(
        query=query,
        report=result.get("report", ""),
        ranked_products=result.get("ranked_products", []),
    )