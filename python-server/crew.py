"""
crew.py
-------
Runs product search directly in Python (no LLM agent in the loop for
search), then hands the raw, guaranteed-verbatim results to a single
CrewAI comparator agent that does the actual reasoning (compare +
recommend).

WHY NOT THREE SEARCH AGENTS ANYMORE:
The previous version used a separate LLM agent per store, each with
one tool. The tool calls always worked and always returned real data
— but CrewAI only forwards a task's *Final Answer text* to downstream
tasks via `context`, not the raw tool output. The 8B search model
would intermittently write a lazy Final Answer like "results are
shown above" or "some products match" instead of literally copying
the tool's output — so the comparator would receive near-empty input
for that store and correctly (from its point of view) report "no
data," even though the store actually had real products the whole
time. This was confirmed directly in the CLI trace: tool output was
always populated, but the agent's Final Answer sometimes wasn't.

Since the search step does no reasoning at all (call one tool, relay
its text), there's no reason to route it through an LLM turn that can
drop data. Calling the tools directly in Python removes that failure
mode completely — the comparator now always receives exactly what
SerpApi returned, with no lossy paraphrase step in between.

Entry point: run_product_search(query) -> str
"""

import os
import litellm
from crewai import Agent, Task, Crew, Process, LLM
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from litellm.exceptions import RateLimitError

# Call the underlying (non-@tool-wrapped) search functions directly.
# .func unwraps CrewAI's @tool decorator so we can call them as plain
# Python functions without going through an agent/LLM turn.
from search_tools import search_store_a, search_store_b, search_store_c

# Groq's API rejects request params it doesn't recognize. This alone
# does NOT fix the issue below, since drop_params only strips
# top-level request params, not fields injected into message dicts.
litellm.drop_params = True

# Groq's free tier has a tight tokens-per-minute (TPM) budget. Retry
# transient RateLimitErrors with exponential backoff rather than
# letting them crash the /flow request.
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


def _llm(model: str | None = None, max_tokens: int = 1600) -> LLM:
    """Groq-backed LLM factory, used only for the comparator agent now."""
    return LLM(
        model=model or os.environ.get("GROQ_MODEL", "groq/llama-3.3-70b-versatile"),
        api_key=os.environ.get("GROQ_API_KEY"),
        max_tokens=max_tokens,
        temperature=0.3,
    )


def _run_searches(query: str) -> dict[str, str]:
    """
    Call all three store search tools directly in Python.

    Returns the raw text each tool produced, unmodified. This is the
    exact text that used to get lost or paraphrased away by the
    per-store search agents — now it goes straight to the comparator
    with no LLM turn in between to drop it.
    """
    return {
        "Store A (Amazon)": search_store_a.func(query),
        "Store B (eBay)": search_store_b.func(query),
        "Store C (Walmart)": search_store_c.func(query),
    }


def build_crew(query: str, search_results: dict[str, str]) -> Crew:
    compare_llm = _llm(os.environ.get("GROQ_MODEL", "groq/llama-3.3-70b-versatile"), max_tokens=1000)

    comparator_agent = Agent(
        role="Product Comparator & Reporter",
        goal="Compare products found across Store A, B, and C and recommend the best option.",
        backstory="You are a meticulous shopping advisor. You never invent products or "
                   "prices that weren't given to you. If a store had no data, you say so "
                   "instead of guessing. If a store's report contains obviously fake "
                   "placeholder data (e.g. generic names like 'Product 1' or made-up domains "
                   "like 'storea.com'), you treat that store as having no real data rather "
                   "than including it in your comparison.",
        llm=compare_llm,
        verbose=True,
        max_iter=1,  # no tools to call — should answer in a single pass
    )

    results_block = "\n\n".join(
        f"{store}:\n{text}" for store, text in search_results.items()
    )

    compare_task = Task(
        description=(
            "Here are the RAW, VERBATIM search results already retrieved from Store A, "
            "Store B, and Store C. Do not call any tools — these results are final and "
            "complete:\n\n"
            f"{results_block}\n\n"
            f"Compare these products by price and relevance to the original query: "
            f"'{query}'. Recommend the single best option and explain why. "
            "If a store's block above says it has no data, treat that store as having no "
            "data — do not invent products for it. If all stores have no data, say clearly "
            "that no products were found rather than fabricating a recommendation. "
            "Any store block containing obviously placeholder/fake data (generic names, "
            "made-up domains) must be treated as 'no data' and excluded from the table. "
            "IMPORTANT: every product link shown above must be preserved and shown in full "
            "in your output — never omit, shorten, or replace a link with just the product "
            "name."
        ),
        expected_output="A markdown comparison table with columns: Store, Product, Price, "
                         "Link, Key Features — where Link is the full clickable URL for each "
                         "product exactly as given above. Follow the table with one clear "
                         "final recommendation that also includes its link (or a clear "
                         "'no products found' statement if all stores were empty).",
        agent=comparator_agent,
    )

    return Crew(
        agents=[comparator_agent],
        tasks=[compare_task],
        process=Process.sequential,
        verbose=True,
    )


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
)
def run_product_search(query: str) -> str:
    """
    Run the three store searches directly in Python (no LLM turn, so
    no risk of an agent summarizing away real data), then let a single
    comparator agent reason over the guaranteed-verbatim results.
    """
    search_results = _run_searches(query)
    crew = build_crew(query, search_results)
    result = crew.kickoff()
    return str(result)