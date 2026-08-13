"""
crew.py
-------
Replaces the old flow.py + agent.py orchestration with a CrewAI crew:

  - 3 search agents (Store A, Store B, Store C), each using its matching
    tool from search_tools.py
  - 1 comparator/reporter agent that takes all three results and produces
    the final comparison + recommendation

Entry point: run_product_search(query) -> str
"""

import os
import litellm
from crewai import Agent, Task, Crew, Process, LLM

from search_tools import search_store_a, search_store_b, search_store_c

# Groq's API rejects request params it doesn't recognize. This alone
# does NOT fix the issue below, since drop_params only strips
# top-level request params, not fields injected into message dicts.
litellm.drop_params = True

# Groq's free tier has a tight tokens-per-minute (TPM) budget, and this
# crew makes several LLM calls per request (one per agent). Rather than
# letting a transient RateLimitError crash the whole /flow request,
# have LiteLLM automatically retry with exponential backoff.
litellm.num_retries = 3

# Known CrewAI bug (crewAIInc/crewAI#5886): newer CrewAI versions
# inject a "cache_breakpoint" field directly into system/user message
# dicts to support Anthropic-style prompt caching — but they do this
# unconditionally, even for providers like Groq that reject unknown
# message properties outright. Since it's baked into the message
# content (not a request param), litellm.drop_params can't strip it.
# Workaround: no-op the function that adds it, until upstream fixes
# this to only apply for Anthropic-compatible providers.
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except ImportError:
    # Older/newer crewai versions may not have this module — safe to
    # skip; if the bug doesn't apply to your version, this is a no-op.
    pass


def _llm() -> LLM:
    """Groq-backed LLM for all agents in the crew."""
    return LLM(
        model=os.environ.get("GROQ_MODEL", "groq/llama-3.3-70b-versatile"),
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def build_crew(query: str) -> Crew:
    llm = _llm()

    store_a_agent = Agent(
        role="Store A Search Specialist",
        goal=f"Find the best matching products for '{query}' in Store A.",
        backstory="You are a focused product researcher who only searches Store A "
                   "and reports exactly what you find, including when Store A has no data.",
        tools=[search_store_a],
        llm=llm,
        verbose=True,
    )

    store_b_agent = Agent(
        role="Store B Search Specialist",
        goal=f"Find the best matching products for '{query}' in Store B.",
        backstory="You are a focused product researcher who only searches Store B "
                   "and reports exactly what you find, including when Store B has no data.",
        tools=[search_store_b],
        llm=llm,
        verbose=True,
    )

    store_c_agent = Agent(
        role="Store C Search Specialist",
        goal=f"Find the best matching products for '{query}' in Store C.",
        backstory="You are a focused product researcher who only searches Store C "
                   "and reports exactly what you find, including when Store C has no data.",
        tools=[search_store_c],
        llm=llm,
        verbose=True,
    )

    comparator_agent = Agent(
        role="Product Comparator & Reporter",
        goal="Compare products found across Store A, B, and C and recommend the best option.",
        backstory="You are a meticulous shopping advisor. You never invent products or "
                   "prices that weren't reported to you. If a store had no data, you say so "
                   "instead of guessing.",
        llm=llm,
        verbose=True,
    )

    task_a = Task(
        description=f"Search Store A for: '{query}'. Report all products found with "
                     f"title, price, and link. If Store A is unavailable or empty, say so plainly.",
        expected_output="A list of products from Store A, each with title, price, and the "
                         "full product link copied exactly as given by the search tool "
                         "(or a clear 'no data' statement).",
        agent=store_a_agent,
    )

    task_b = Task(
        description=f"Search Store B for: '{query}'. Report all products found with "
                     f"title, price, and link. If Store B is unavailable or empty, say so plainly.",
        expected_output="A list of products from Store B, each with title, price, and the "
                         "full product link copied exactly as given by the search tool "
                         "(or a clear 'no data' statement).",
        agent=store_b_agent,
    )

    task_c = Task(
        description=f"Search Store C for: '{query}'. Report all products found with "
                     f"title, price, and link. If Store C is unavailable or empty, say so plainly.",
        expected_output="A list of products from Store C, each with title, price, and the "
                         "full product link copied exactly as given by the search tool "
                         "(or a clear 'no data' statement).",
        agent=store_c_agent,
    )

    compare_task = Task(
        description="Using the results from Store A, Store B, and Store C, compare the "
                     "products by price and relevance to the original query: "
                     f"'{query}'. Recommend the single best option and explain why. "
                     "If all stores returned no data, say clearly that no products were found "
                     "rather than fabricating a recommendation. "
                     "IMPORTANT: every product link from the store reports must be preserved "
                     "and shown in full in your output — never omit, shorten, or replace a "
                     "link with just the product name.",
        expected_output="A markdown comparison table with columns: Store, Product, Price, "
                         "Link, Key Features — where Link is the full clickable URL for each "
                         "product exactly as reported by the store agents. Follow the table "
                         "with one clear final recommendation that also includes its link "
                         "(or a clear 'no products found' statement if all stores were empty).",
        agent=comparator_agent,
        context=[task_a, task_b, task_c],
    )

    return Crew(
        agents=[store_a_agent, store_b_agent, store_c_agent, comparator_agent],
        tasks=[task_a, task_b, task_c, compare_task],
        process=Process.sequential,
        verbose=True,
    )


def run_product_search(query: str) -> str:
    """Kick off the crew for a given query and return the final report text."""
    crew = build_crew(query)
    result = crew.kickoff()
    return str(result)