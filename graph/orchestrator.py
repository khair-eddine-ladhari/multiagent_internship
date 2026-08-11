"""
Main entry point for the pipeline. No graph, no compiled object --
just a plain function that classifies the query, then dispatches to
the right agent function with a simple if/elif chain.
"""

from agents.nodes import (
    classify_intent,
    run_chat_agent,
    search_source_a,
    search_source_b,
    normalize_results,
    rank_results,
    build_report,
)


def run_orchestrator(query: str) -> dict:
    query = query.strip()

    intent = classify_intent(query)

    if intent == "chat":
        return {
            "intent": "chat",
            "report": run_chat_agent(query),
        }

    if intent == "clarify":
        return {
            "intent": "clarify",
            "needs_clarification": True,
            "clarification_question": (
                f"Your query \"{query}\" is a bit broad -- "
                "could you share a budget or specific features you want?"
            ),
        }

    # intent == "search"
    raw_results = search_source_a(query) + search_source_b(query)
    normalized = normalize_results(raw_results)

    if not normalized:
        return {
            "intent": "search",
            "report": f'No results found for "{query}".',
            "ranked_products": [],
        }

    ranked = rank_results(normalized)
    report = build_report(query, ranked)

    return {
        "intent": "search",
        "report": report,
        "ranked_products": ranked,
    }