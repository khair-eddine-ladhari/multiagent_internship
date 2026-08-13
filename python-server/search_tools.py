"""
search_tools.py
----------------
Real product search tools backed by SerpApi (https://serpapi.com),
which offers dedicated engines for Amazon, eBay, and Walmart with a
single API key and a consistent JSON interface — no per-store
developer approval required.

Requires SERPAPI_KEY to be set in the environment (see .env).

Each function returns a short plain-text summary (title, price, link
per result) because these tools feed into CrewAI agents, which work
with text, not structured objects.
"""

import os
import requests
from crewai.tools import tool

SERPAPI_ENDPOINT = "https://serpapi.com/search"
RESULTS_PER_STORE = 3  # fewer results = fewer tokens sent to the LLM per agent call


def _serpapi_search(engine: str, query_param: str, query: str) -> list[dict]:
    """
    Call SerpApi for a given engine (amazon / ebay / walmart) and
    return its list of organic results (raw dicts), or [] on any
    failure — callers turn that into a "no data" message rather than
    raising, so one store's outage doesn't break the whole crew run.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        return []

    params = {
        "engine": engine,
        "api_key": api_key,
        query_param: query,
    }

    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    return data.get("organic_results", [])[:RESULTS_PER_STORE]


def _format_results(store_name: str, query: str, results: list[dict]) -> str:
    """Turn a list of SerpApi result dicts into a short text report for the agent."""
    if not results:
        return f"{store_name} has no data for '{query}' at this time."

    lines = [f"{store_name} results for '{query}':"]
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled product")
        if len(title) > 80:
            title = title[:77] + "..."

        # Price field names differ slightly between SerpApi engines.
        price = (
            item.get("price")
            or item.get("price_raw")
            or item.get("primary_offer", {}).get("offer_price")
            or "price not listed"
        )

        link = item.get("link") or item.get("product_page_url") or "no link available"

        lines.append(f"{i}. {title} - {price} - {link}")

    return "\n".join(lines)


@tool("search_store_a")
def search_store_a(query: str) -> str:
    """Search Amazon for products matching the query. Returns title, price, and link for each match."""
    results = _serpapi_search(engine="amazon", query_param="k", query=query)
    return _format_results("Amazon (Store A)", query, results)


@tool("search_store_b")
def search_store_b(query: str) -> str:
    """Search eBay for products matching the query. Returns title, price, and link for each match."""
    results = _serpapi_search(engine="ebay", query_param="_nkw", query=query)
    return _format_results("eBay (Store B)", query, results)


@tool("search_store_c")
def search_store_c(query: str) -> str:
    """Search Walmart for products matching the query. Returns title, price, and link for each match."""
    results = _serpapi_search(engine="walmart", query_param="query", query=query)
    return _format_results("Walmart (Store C)", query, results)