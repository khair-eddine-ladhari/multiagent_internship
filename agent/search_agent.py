"""
Search agent (specialist, owned by the top-level orchestrator).

This agent's only job: given a product query, decide WHICH source(s) to
search (source A, source B, Pinecone, or some combination) and call
them. It doesn't rank or write the report -- that's ranking_agent.py
and report_generator.py.

The underlying tool functions live in tools/*.py as plain, testable
functions. Here they're wrapped with @tool so the LLM can see their
name/description/schema and decide which to call -- the wrapping stays
here (not in tools/) so tools/ functions can still be called directly
and tested without pulling in LangChain at all.
"""

import os
import json

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from tools.source_a_tools import search_source_a as _search_source_a
from tools.source_b_tools import search_source_b as _search_source_b
from tools.pinecone_tools import query_pinecone as _query_pinecone


SYSTEM_PROMPT = """You are a search planning agent for a product search system.

Given a user's product query, decide which source(s) to search:

- search_source_a: a live external product source. Use for most queries
  needing current, real-world product listings.
- search_source_b: a second, independent external product source. Use
  alongside source A for broader coverage, or alone if source A already
  covers this well and you want a different source's listings too.
- query_pinecone: searches previously indexed/cached product data. Use
  when the query might match something already searched before, or to
  reduce redundant external calls.

You may call one tool, multiple tools, or the same tool with a refined
query if the first results seem insufficient. Call tools as needed,
then stop once you have enough results to hand off for ranking.
"""


@tool
def search_source_a(query: str) -> str:
    """Search source A (external product listings) for the given query.
    Returns a JSON list of {title, price, url, source} dicts."""
    try:
        results = _search_source_a(query)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(results)


@tool
def search_source_b(query: str) -> str:
    """Search source B (a second, independent external product source)
    for the given query. Returns a JSON list of {title, price, url, source} dicts."""
    try:
        results = _search_source_b(query)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(results)


@tool
def query_pinecone(query: str) -> str:
    """Search previously indexed/cached product data for the given query.
    Returns a JSON list of {id, score, metadata} dicts."""
    try:
        results = _query_pinecone(query)
    except (RuntimeError, NotImplementedError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(results)


TOOLS = [search_source_a, search_source_b, query_pinecone]


def run_search_agent(query: str) -> list[dict]:
    """
    Runs the search agent loop: the LLM decides which tool(s) to call,
    tools execute, results are collected until the LLM stops calling
    tools. Returns the combined raw product results from every tool
    call that succeeded (errors/unconfigured tools are skipped, not
    raised, so one broken source doesn't kill the whole search).
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    tool_map = {t.name: t for t in TOOLS}
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", query),
    ]

    raw_results: list[dict] = []

    # Cap iterations so a confused LLM can't loop forever.
    for _ in range(4):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            tool_fn = tool_map[call["name"]]
            output = tool_fn.invoke(call["args"])

            try:
                parsed = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                parsed = []

            if isinstance(parsed, list):
                raw_results.extend(parsed)
            # dicts with "error" are silently skipped -- unconfigured
            # or failed sources shouldn't break the whole search.

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": output,
            })

    return raw_results