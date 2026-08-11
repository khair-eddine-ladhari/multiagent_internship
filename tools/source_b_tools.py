"""
Tool: search source B -- eBay, via SerpAPI.

Same provider as source A (SerpAPI) but a different engine, so A and B
give genuinely independent listings rather than duplicating each other.
Requires SERPAPI_API_KEY in .env. Docs: https://serpapi.com/ebay-search-api
"""

import os
import requests

BASE_URL = "https://serpapi.com/search"
API_KEY = os.environ.get("SERPAPI_API_KEY")


def search_source_b(query: str, limit: int = 5) -> list[dict]:
    """
    Search eBay (via SerpAPI) for products matching `query`.

    Returns a list of dicts shaped like:
        {"source": "source_b", "title": str, "price": float, "url": str}

    Raises:
        RuntimeError if SERPAPI_API_KEY isn't set, or if the request fails.
    """
    if not API_KEY:
        raise RuntimeError(
            "source_b_tools is not configured yet -- set SERPAPI_API_KEY "
            "in .env before calling search_source_b()."
        )

    params = {
        "engine": "ebay",
        "_nkw": query,
        "api_key": API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic_results", [])[:limit]:
        price_raw = item.get("price", {})
        price_value = price_raw.get("extracted") if isinstance(price_raw, dict) else price_raw
        results.append({
            "source": "source_b",
            "title": item.get("title", ""),
            "price": float(price_value) if price_value else 0.0,
            "url": item.get("link", ""),
        })

    return results