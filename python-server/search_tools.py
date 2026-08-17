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

import logging

logger = logging.getLogger("search_tools")
logging.basicConfig(level=logging.INFO)


SERPAPI_ENDPOINT = "https://serpapi.com/search"
RESULTS_PER_STORE = 3  # fewer results = fewer tokens sent to the LLM per agent call



def _serpapi_search(engine: str, query_param: str, query: str) -> list[dict]:
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        logger.warning("SERPAPI_KEY not set — returning no results for %s", engine)
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
    except requests.RequestException as e:
        logger.warning("SerpApi request failed for engine=%s: %s", engine, e)
        return []
    except ValueError as e:
        logger.warning("SerpApi returned invalid JSON for engine=%s: %s", engine, e)
        return []

    if "error" in data:
        logger.warning("SerpApi returned an error for engine=%s: %s", engine, data["error"])
        return []

    results = data.get("organic_results", [])
    if not results:
        logger.info("SerpApi engine=%s returned 0 organic_results for query=%r. Keys: %s",
                     engine, query, list(data.keys()))

    return results[:RESULTS_PER_STORE]


def _normalize_price(item: dict) -> str:
    """
    Return a clean, LLM-safe price string regardless of which SerpApi
    engine produced the result.

    Price field shapes vary by engine and are NOT always plain strings:
      - Amazon:  item["price"] is usually already a string, e.g. "$8.99"
      - eBay:    item["price"] is a dict, e.g. {"raw": "$21.74", "extracted": 21.74}
      - Walmart: item["primary_offer"]["offer_price"] is a float, e.g. 24.99

    Previously, `item.get("price") or ...` would return the eBay dict
    as-is (since a non-empty dict is truthy), and str()-interpolating
    that dict into the report text produced garbled input like
    "{'raw': '$21.74', 'extracted': 21.74}" — which was confusing
    enough that the 8B search-agent model sometimes gave up and
    reported "no products found" even though real data was present.

    This function checks the *type* of each candidate field, not just
    truthiness, so a dict or float is converted to a proper string
    instead of ever being passed through raw.
    """
    price = item.get("price")
    if isinstance(price, str) and price.strip():
        return price.strip()
    if isinstance(price, dict):
        raw = price.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        extracted = price.get("extracted")
        if isinstance(extracted, (int, float)):
            return f"${extracted}"

    price_raw = item.get("price_raw")
    if isinstance(price_raw, str) and price_raw.strip():
        return price_raw.strip()

    offer = item.get("primary_offer")
    if isinstance(offer, dict):
        offer_price = offer.get("offer_price")
        if isinstance(offer_price, str) and offer_price.strip():
            return offer_price.strip()
        if isinstance(offer_price, (int, float)):
            return f"${offer_price}"

    return "price not listed"


def _clean_link(link: str) -> str:
    """
    Strip tracking/session cruft from product links before they ever
    reach the LLM.

    eBay's `link` field from SerpApi is the raw item URL *plus* a huge
    `itmprp=enc%3A...` tracking blob (plus `_skw`, `itmmeta`, `hash`,
    etc.), often several hundred characters long. That's not just
    wasted tokens — it's a direct contributor to output getting
    truncated mid-URL, since the blob alone can eat a meaningful chunk
    of max_tokens once it's echoed back in the comparator's table.

    eBay item URLs are of the form:
        https://www.ebay.com/itm/<item_id>?<tracking params...>
    Everything after the item ID is tracking/session data, not needed
    to reach the listing. So for ebay.com links, rebuild just
    `https://www.ebay.com/itm/<item_id>` and drop the query string
    entirely. Any other domain (Amazon, Walmart) is returned unchanged,
    since their links don't carry this bloat.
    """
    if not isinstance(link, str) or not link:
        return link

    if "ebay.com/itm/" in link:
        # Item ID is the run of digits immediately after "itm/",
        # before the query string starts.
        after = link.split("ebay.com/itm/", 1)[1]
        item_id = ""
        for ch in after:
            if ch.isdigit():
                item_id += ch
            else:
                break
        if item_id:
            return f"https://www.ebay.com/itm/{item_id}"

    return link


def _format_results(store_name: str, query: str, results: list[dict]) -> str:
    """Turn a list of SerpApi result dicts into a short text report for the agent."""
    if not results:
        return f"{store_name} has no data for '{query}' at this time."

    lines = [f"{store_name} results for '{query}':"]
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled product")
        if len(title) > 80:
            title = title[:77] + "..."

        price = _normalize_price(item)

        link = item.get("link") or item.get("product_page_url") or "no link available"
        link = _clean_link(link)

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