"""
Report generator (specialist, owned by the top-level orchestrator).

Last step in the pipeline. No tools -- by this point search_agent has
already found products and ranking_agent has already ordered them by
price + quality. This agent's only job is synthesis: turn that ranked
list into a comprehensive, readable recommendation report.

This is intentionally an LLM call (not string formatting) because the
brief asks for a "comprehensive recommendation report" -- i.e. an
actual explanation of tradeoffs (why the top pick is the top pick,
when a cheaper option might still make sense), not just a re-listing
of the data the user can already see in ranked_products.
"""

import os
import json

from langchain_groq import ChatGroq


SYSTEM_PROMPT = """You are a product recommendation report writer.

You will be given the user's original query and a JSON list of
products already ranked best-to-worst (price and quality both
considered), each with title, price, url, source, average_rating,
and review_count.

Write a comprehensive but concise recommendation report:
- Lead with the top recommendation and WHY it's the top pick (price
  vs. quality reasoning), not just a repeat of its stats.
- Briefly cover the next 1-2 alternatives and when someone might
  prefer them instead (e.g. "if budget is the main concern, X is
  cheaper but has a lower rating").
- Keep it grounded in the data given -- do not invent products,
  prices, or ratings not present in the list.
- Plain text, no markdown headers. A few short paragraphs is enough.
"""


def run_report_generator(query: str, ranked_products: list[dict]) -> str:
    """
    Synthesizes a recommendation report from the ranked product list.

    Returns a plain-text report. Falls back to a simple templated
    summary if the LLM call fails, so the pipeline still returns
    something useful instead of erroring out at the last step.
    """
    if not ranked_products:
        return f'No results found for "{query}".'

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.3,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", json.dumps({"query": query, "ranked_products": ranked_products})),
    ]

    try:
        response = llm.invoke(messages)
        if response.content.strip():
            return response.content.strip()
    except Exception:
        pass

    # Fallback: plain templated summary, so a failed LLM call doesn't
    # mean the user gets nothing back after search + ranking succeeded.
    lines = [f'Results for "{query}":']
    for item in ranked_products:
        rating = item.get("average_rating", 0)
        lines.append(
            f"- {item.get('title', 'Unknown')} (${item.get('price', 0)}, "
            f"{rating}★, via {item.get('source', 'unknown')})"
        )
    return "\n".join(lines)