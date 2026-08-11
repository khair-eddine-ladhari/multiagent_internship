"""
Query understanding agent (specialist, owned by the top-level orchestrator).

First step in the pipeline. Decides:
  1. Is this casual chat, or an actual product search request?
  2. If it's a search request, is it specific enough to search on, or
     does it need clarification first?

This mirrors the classify_intent() pattern from the old project
(short-circuit obvious cases without an LLM call, fall back to the LLM
for everything else) -- kept as one LLM call doing both jobs at once
rather than two separate calls, since they're closely related and
cheap to combine.
"""

import os
import json

from langchain_groq import ChatGroq


GREETINGS = {"hi", "hello", "hey", "yo", "thanks", "thank you", "ok", "okay", "sup", "hiya"}

SYSTEM_PROMPT = """You classify incoming messages for a product search assistant.

Respond with ONLY a JSON object, no other text:
{"intent": "chat" | "search", "needs_clarification": true | false, "clarification_question": "..." | null}

Rules:
- "chat": greetings, small talk, or anything not a product search request.
  needs_clarification is always false, clarification_question is null.
- "search": the user wants to find/compare a product. If the query is
  specific enough to actually search on (has a product type, and
  ideally a budget or key feature), set needs_clarification to false.
  If it's too vague (e.g. just "laptop" with no budget or use case),
  set needs_clarification to true and write a short, specific
  clarification_question asking for what's missing (budget, use case,
  brand preference, etc).
"""


def run_query_understanding(query: str) -> dict:
    """
    Returns a dict shaped like:
        {"intent": "chat" | "search", "needs_clarification": bool,
         "clarification_question": str | None}

    Short-circuits obvious greetings without an LLM call, same
    reasoning as the old project: classification isn't perfectly
    deterministic, so trivial cases should always resolve the same way.
    """
    stripped = query.strip().lower()

    if stripped in GREETINGS or not stripped:
        return {"intent": "chat", "needs_clarification": False, "clarification_question": None}

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", query),
    ]

    try:
        response = llm.invoke(messages)
        result = json.loads(response.content)
        if result.get("intent") in ("chat", "search"):
            return {
                "intent": result["intent"],
                "needs_clarification": bool(result.get("needs_clarification", False)),
                "clarification_question": result.get("clarification_question"),
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: if classification fails outright, treat as a search
    # query needing clarification rather than silently guessing.
    return {
        "intent": "search",
        "needs_clarification": True,
        "clarification_question": "Could you clarify what you're looking for?",
    }