"""
Tool: search source A -- Google Shopping, via SerpAPI.

Requires SERPAPI_API_KEY in .env. Docs: https://serpapi.com/google-shopping-api
"""

import os
import requests

BASE_URL = "https://serpapi.com/search"
API_KEY = os.environ.get("SERPAPI_API_KEY")


def search_source_a(query: str, limit: int = 5) -> list[dict]:
    """
    Search Google Shopping (via SerpAPI) for products matching `query`.

    Returns a list of dicts shaped like:
        {"source": "source_a", "title": str, "price": float, "url": str}

    Raises:
        RuntimeError if SERPAPI_API_KEY isn't set, or if the request fails.
    """
    if not API_KEY:
        raise RuntimeError(
            "source_a_tools is not configured yet -- set SERPAPI_API_KEY "
            "in .env before calling search_source_a()."
        )

    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("shopping_results", [])[:limit]:
        price_raw = item.get("extracted_price", item.get("price", 0))
        results.append({
            "source": "source_a",
            "title": item.get("title", ""),
            "price": float(price_raw) if price_raw else 0.0,
            "url": item.get("product_link", item.get("link", "")),
        })

    return results