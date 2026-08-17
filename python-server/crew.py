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
import re
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


def _llm(model: str | None = None, max_tokens: int = 2200) -> LLM:
    """
    Groq-backed LLM factory, used only for the comparator agent now.

    Model: qwen/qwen3.6-27b. llama-3.3-70b-versatile is genuinely
    deprecated on this account now (confirmed via a live
    litellm.exceptions.NotFoundError: "model_not_found" — an earlier
    successful run was likely from before the deprecation took effect,
    or from a different session). qwen3.6-27b was chosen over Groq's
    other suggested replacement, openai/gpt-oss-120b, because the
    "openai/" substring in that model's name triggers a LiteLLM
    provider-detection bug that misroutes requests to OpenAI's own API
    (BerriAI/litellm#14807) — qwen3.6-27b has no such naming collision.

    max_tokens is intentionally NOT being raised further to fix
    truncation. This account is on Groq's free tier with a tight TPM
    budget, and every extra token allowed here is an extra token that
    can trip the rate limit on the very next call. The truncation seen
    with a 9-row table + open-ended recommendation text was fixed by
    constraining the recommendation's length in the task prompt below
    (see compare_task), not by growing this ceiling. Only raise this
    number if the table itself (not the recommendation) is getting
    cut off after confirming eBay links are already being cleaned by
    search_tools._clean_link().
    """
    return LLM(
        model=model or os.environ.get("GROQ_MODEL", "groq/qwen/qwen3.6-27b"),
        api_key=os.environ.get("GROQ_API_KEY"),
        max_tokens=max_tokens,
        temperature=0.3,
        reasoning_format="hidden",
        # qwen3 models specifically support fully disabling reasoning via
        # reasoning_effort="none" (unlike gpt-oss, where reasoning is
        # always-on and can only be hidden, not skipped). If this is
        # honored, there's no <think> trace generated at all, so
        # max_tokens can stay much lower than a reasoning-enabled budget
        # would require. _strip_thinking() below still runs regardless,
        # as a safety net in case this parameter also doesn't get
        # reliably forwarded by CrewAI's LLM wrapper.
        reasoning_effort="none",
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
    compare_llm = _llm(os.environ.get("GROQ_MODEL", "groq/qwen/qwen3.6-27b"))

    comparator_agent = Agent(
        role="Product Comparator & Reporter",
        goal="Compare products found across Store A, B, and C and recommend the best option.",
        backstory="You are a meticulous shopping advisor. You never invent products or "
                   "prices that weren't given to you. If a store had no data, you say so "
                   "instead of guessing. If a store's report contains obviously fake "
                   "placeholder data (e.g. generic names like 'Product 1' or made-up domains "
                   "like 'storea.com'), you treat that store as having no real data rather "
                   "than including it in your comparison. You write concisely — you never "
                   "pad your answer with numbered lists or multi-paragraph justifications "
                   "when a couple of sentences will do.",
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
            "name. "
            "IMPORTANT: keep your final recommendation explanation to 2-3 sentences of "
            "plain prose. Do NOT use a numbered or bulleted list for the explanation, and "
            "do not write more than one short paragraph — the table above is the detailed "
            "part of the answer, the recommendation is just a brief pointer to the best pick."
        ),
        expected_output="A markdown comparison table with columns: Store, Product, Price, "
                         "Link, Key Features — where Link is the full clickable URL for each "
                         "product exactly as given above. Follow the table with ONE short "
                         "paragraph (2-3 sentences, no lists) naming the single best pick, "
                         "its link, and a brief reason why (or a clear 'no products found' "
                         "statement if all stores were empty).",
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
    return _strip_thinking(str(result))


def _strip_thinking(text: str) -> str:
    """
    Remove any visible <think>...</think> reasoning block from the model's
    output.

    qwen3.6-27b is a reasoning model that, by default, writes its full
    chain-of-thought into the response before the actual answer. The
    reasoning_format="hidden" parameter is supposed to suppress this at
    the API level, but CrewAI's LLM wrapper doesn't reliably forward all
    non-standard kwargs to LiteLLM in every code path (same class of bug
    as the earlier base_url/api_base issue) — reasoning_format="hidden"
    was still showing up in the raw output during testing. This is a
    defense-in-depth fix at the Python level that works regardless of
    whether the API-level parameter actually took effect: if a <think>
    block is present, drop it and return only what follows.
    """
    if "<think>" not in text:
        return text.strip()

    # If the closing tag is missing (response got cut off mid-thought,
    # e.g. from an insufficient max_tokens budget on a previous run),
    # there's no real answer to recover — surface that clearly rather
    # than returning an empty string that looks like a silent failure.
    if "</think>" not in text:
        return (
            "The model's response was cut off while still reasoning and "
            "never produced a final answer. Try again, or increase "
            "max_tokens further in crew.py if this keeps happening."
        )

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()