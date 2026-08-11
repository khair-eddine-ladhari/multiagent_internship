"""
Ranking agent (specialist, owned by the top-level orchestrator).

This agent's only job: given a list of normalized product results,
enrich them with rating data and produce a ranked order that weighs
price AND quality -- not just cheapest-first. It doesn't search or
write the report -- that's search_agent.py and report_generator.py.

Two phases:
  1. Deterministic: fetch ratings for every product via rating_tools
     (no LLM needed here -- it's a lookup, not a decision).
  2. LLM reasoning: given price + rating data together, the LLM decides
     the final order and returns it as structured JSON. This is the
     "quality vs price tradeoff" the project brief asks for -- a
     $700 product with 4.8 stars might reasonably rank above a $650
     product with 3.1 stars, which a pure price-sort would miss.
"""

import os
import json

from langchain_groq import ChatGroq

from tools.rating_tools import get_ratings_batch


SYSTEM_PROMPT = """You are a product ranking agent.

You will be given a JSON list of products, each with a title, price,
url, source, and rating info (average_rating, review_count). Rank them
from best to worst, weighing BOTH price and quality -- not price alone.

Guidelines:
- A notably higher rating can outweigh a modest price difference.
- A product with review_count of 0 has no rating signal -- treat it as
  neutral, not bad, and rely more on price for that item.
- Ties or close calls: prefer the cheaper option.

Respond with ONLY a JSON list of the same product objects, reordered
best-to-worst. Do not add commentary, do not change the objects' fields.
"""


def _enrich_with_ratings(products: list[dict]) -> list[dict]:
    """Attaches rating info to each product via a batch lookup."""
    urls = [p.get("url", "") for p in products]
    ratings = get_ratings_batch(urls)

    enriched = []
    for product in products:
        rating = ratings.get(product.get("url", ""), {"average_rating": 0.0, "review_count": 0})
        enriched.append({**product, **rating})

    return enriched


def run_ranking_agent(products: list[dict]) -> list[dict]:
    """
    Ranks `products` (already normalized) by price and quality together.

    Returns the same list, reordered best-to-worst. Falls back to a
    plain price-ascending sort if the LLM call fails or returns
    something unparseable, so ranking never crashes the pipeline.
    """
    if not products:
        return []

    enriched = _enrich_with_ratings(products)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", json.dumps(enriched)),
    ]

    try:
        response = llm.invoke(messages)
        ranked = json.loads(response.content)
        if isinstance(ranked, list) and ranked:
            return ranked
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: price ascending, so ranking always returns *something*.
    return sorted(enriched, key=lambda item: item.get("price", 0))