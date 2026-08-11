"""
Normalization (specialist, owned by the top-level orchestrator).

Takes the combined raw results from search_agent (which may come from
source A, source B, and/or Pinecone -- each with slightly different
field shapes) and standardizes them into one consistent shape before
ranking_agent sees them.

No LLM needed here -- this is deterministic data cleaning, not a
judgment call. Kept as its own file (rather than folded into
search_agent) so ranking_agent and report_generator can always assume
one consistent product shape regardless of which source(s) contributed.
"""


REQUIRED_FIELDS = ("source", "title", "price", "url")


def _coerce_price(price) -> float:
    """Handles price arriving as a number, a numeric string, or a
    currency-formatted string like "$799.99" -- returns 0.0 if it
    can't be parsed rather than raising, so one bad record doesn't
    break the whole batch."""
    if isinstance(price, (int, float)):
        return float(price)

    if isinstance(price, str):
        cleaned = price.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    return 0.0


def normalize_results(raw_results: list[dict]) -> list[dict]:
    """
    Standardizes a list of raw product dicts (possibly from multiple
    sources with different shapes) into a consistent shape:
        {"source": str, "title": str, "price": float, "url": str}

    Drops entries missing a title (nothing useful to show the user),
    and skips Pinecone-style matches (id/score/metadata) that haven't
    been mapped to product fields yet, rather than passing through a
    shape ranking_agent/report_generator don't expect.
    """
    normalized = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        # Pinecone matches come back as {id, score, metadata} rather
        # than product fields directly -- pull product data out of
        # metadata if present, otherwise skip (nothing to show yet).
        if "metadata" in item and "title" not in item:
            item = item.get("metadata", {})

        title = item.get("title", "").strip()
        if not title:
            continue

        normalized.append({
            "source": item.get("source", "unknown"),
            "title": title,
            "price": _coerce_price(item.get("price", 0)),
            "url": item.get("url", ""),
        })

    return normalized