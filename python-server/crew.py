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

FOLLOW-UP / HISTORY HANDLING:
main.py stores each turn's query+answer in session_state and passes
it back on every request, but the search itself was always run on the
raw current-turn query alone. That meant a follow-up like "which one
has the best battery life" — meant to refer back to the laptops from
the previous turn — got searched literally, returning AA batteries
and car batteries instead of laptop battery life.

_resolve_query() fixes this with a single, cheap, non-agentic LLM
call: given the last turn (query + answer) and the new question, it
either rewrites the new question into a standalone query that names
the single recommended prior product, or returns it unchanged if it
wasn't actually a follow-up. This keeps the fix isolated to one small
step — the store-search tools and the comparator agent are both
otherwise untouched.

COMPARATOR RELEVANCE CHECK:
Rewriting the query correctly (e.g. to "ASUS Chromebook CX15 battery
life") does NOT guarantee the store search tools return anything
about battery life — they do plain keyword matching, so a query like
that can return replacement-battery accessories, wrong sub-models, or
full laptop listings with no spec info at all. Left unchecked, the
comparator agent would quietly fall back to comparing on price (or
whatever data it did have) and present that as if it answered the
original question — which is a worse failure than saying "I don't
know," because it looks like a real answer.

The fix: the comparator task requires a literal, greppable line in
the output — "Relevant data found: yes" or "Relevant data found: no"
— BEFORE any recommendation text, and explicit instructions that a
"no" must not be followed by a recommendation on an unrelated basis.
This is more reliable on a small free-tier model than asking it to
"check first" as pure prose, and it gives calling code a concrete
string to assert on / log, rather than having to parse full prose to
detect a silent substitution. It's still just a prompt instruction
though, so _sanity_check_relevance_claim() below adds a second,
independent, non-LLM check on top of it (see that function's
docstring).

DETERMINISTIC GUARDS (added on top of the prompt-only version):
Three of the behaviors above were originally *only* prompt
instructions to the comparator LLM — which means a bad response from
the (free-tier, small) model could silently violate them with nothing
to catch it. These are now enforced in plain Python before/around the
LLM call, so they no longer depend on the model reliably following
instructions:
  - _all_empty()              -> all-stores-empty case
  - _looks_like_placeholder() -> fake/placeholder listing detection
  - _sanity_check_relevance_claim() -> catches a "yes" that isn't
    actually backed by any query-relevant text in the output

Entry point: run_product_search(query, history=None) -> str
"""

import logging
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

logger = logging.getLogger(__name__)

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

GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/qwen/qwen3.6-27b")

# Cap how much prior answer text (e.g. a full 9-row markdown table) we
# feed into the rewrite prompt. We only need enough for the model to
# identify which product "which one" refers to, not the whole table
# verbatim — and staying small matters on Groq's free-tier TPM budget.
_MAX_PRIOR_ANSWER_CHARS = 1200

# Patterns that indicate a store's search text is empty / no results,
# used by _all_empty() so the "no products found" case is a
# deterministic Python check instead of something the comparator LLM
# has to notice and report correctly on its own.
_NO_RESULTS_RE = re.compile(
    r"\bno (products?|results?|items?|listings?|matches)\s+found\b"
    r"|\bnothing found\b"
    r"|\bno matches\b",
    re.IGNORECASE,
)

# Heuristics for obviously fake/placeholder listing text (generic
# names, made-up domains). Used by _looks_like_placeholder() so this
# is a Python-level filter applied before the comparator ever sees
# the data, rather than an instruction the model has to remember to
# apply while writing its table.
_PLACEHOLDER_NAME_RE = re.compile(r"\b(product|item)\s*[#\-]?\s*\d+\b", re.IGNORECASE)
_PLACEHOLDER_DOMAIN_RE = re.compile(
    r"\b(store\s*[abc]|example|test|placeholder)\.(com|net|org)\b", re.IGNORECASE
)


def _llm(model: str | None = None, max_tokens: int = 2200) -> LLM:
    """
    Groq-backed LLM factory, used for the comparator agent and the
    query-rewrite step.

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
        model=model or GROQ_MODEL,
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


def _all_empty(search_results: dict[str, str]) -> bool:
    """
    Deterministic, Python-level check for "every store came back
    empty," so this no longer depends on the comparator LLM noticing
    and saying so in prose. A store counts as empty if its text is
    blank or matches a "no results" pattern; if any single store has
    real content, this returns False.
    """
    for text in search_results.values():
        stripped = (text or "").strip()
        if stripped and not _NO_RESULTS_RE.search(stripped):
            return False
    return True


def _looks_like_placeholder(text: str) -> bool:
    """
    Heuristic check for obviously fake/placeholder listing text —
    generic names like "Product 1" or made-up domains like
    "storea.com". This is intentionally crude (it flags the whole
    store block, not individual listings) — the goal is to catch the
    obvious dev/test-data case, not to be a general fraud detector.
    """
    if not text:
        return False
    return bool(_PLACEHOLDER_NAME_RE.search(text) or _PLACEHOLDER_DOMAIN_RE.search(text))


def _filter_placeholder_results(search_results: dict[str, str]) -> dict[str, str]:
    """
    Replace any store's text that looks like placeholder/fake data
    with an explicit "no real data" marker before it ever reaches the
    comparator prompt. Previously this relied entirely on the
    comparator LLM noticing and excluding it itself.
    """
    filtered = {}
    for store, text in search_results.items():
        if _looks_like_placeholder(text):
            logger.info("Filtered placeholder/fake data from %s", store)
            filtered[store] = "No real data found (search returned placeholder/fake listings)."
        else:
            filtered[store] = text
    return filtered


def _resolve_query(query: str, history: list[dict] | None) -> str:
    """
    Resolve a possibly-context-dependent follow-up query into a
    standalone one, using only the most recent prior turn.

    Examples:
      history has a turn about laptops, query = "which one has the
      best battery life" -> "ASUS Chromebook CX15 battery life"

      history has a turn about laptops, query = "show me headphones
      under $50" -> unchanged (not actually a follow-up)

    If there's no history, or the rewrite call fails for any reason,
    fall back to the original query untouched rather than blocking
    the whole search on a rewrite failure.
    """
    if not history:
        return query

    last_turn = history[-1]
    prior_query = last_turn.get("query", "")
    prior_answer = last_turn.get("answer", "")[:_MAX_PRIOR_ANSWER_CHARS]

    if not prior_query and not prior_answer:
        return query

    rewrite_prompt = (
        "You rewrite follow-up shopping questions into standalone search queries.\n\n"
        f"Previous user query: {prior_query}\n"
        f"Previous answer (may be truncated):\n{prior_answer}\n\n"
        f'New question: "{query}"\n\n'
        "If the new question refers back to something from the previous answer (e.g. it "
        "says \"which one\", \"that one\", \"the first option\", \"is it good for X\", or "
        "otherwise doesn't make sense as a search on its own):\n"
        "  - The previous answer ends with a single recommended product — use THAT one "
        "product, not the full list of products mentioned in the table.\n"
        "  - Rewrite the new question into a short, standalone search query naming just "
        "that one specific product plus what's being asked about it "
        "(e.g. \"ASUS Chromebook CX15 battery life\", NOT a list of every product "
        "compared against each other).\n"
        "  - Never join multiple product names together with \"vs\" — pick the single "
        "recommended one only.\n"
        "If the new question does NOT refer back to anything above and stands fine on its "
        "own, return it completely unchanged.\n\n"
        "Examples:\n"
        "  Previous query: \"best budget laptops\"\n"
        "  Previous answer ends with: \"...I'd recommend the ASUS Chromebook CX15.\"\n"
        "  New question: \"which one has the best battery life\"\n"
        "  -> \"ASUS Chromebook CX15 battery life\"\n\n"
        "  Previous query: \"best budget laptops\"\n"
        "  Previous answer ends with: \"...I'd recommend the ASUS Chromebook CX15.\"\n"
        "  New question: \"show me headphones under $50\"\n"
        "  -> \"show me headphones under $50\" (unchanged — not a follow-up)\n\n"
        "Reply with ONLY the resulting search query text — no quotes, no explanation, no "
        "extra words."
    )

    try:
        rewrite_llm = _llm(GROQ_MODEL, max_tokens=120)
        response = rewrite_llm.call(messages=[{"role": "user", "content": rewrite_prompt}])
        rewritten = _strip_thinking(str(response)).strip().strip('"')
        return rewritten or query
    except Exception:
        # Rewrite is a best-effort enhancement, not a hard dependency —
        # if Groq is rate-limited or errors here, just search the raw
        # query rather than failing the whole request.
        logger.warning("Query rewrite failed, falling back to raw query", exc_info=True)
        return query


def build_crew(query: str, search_results: dict[str, str]) -> Crew:
    compare_llm = _llm(GROQ_MODEL)

    comparator_agent = Agent(
        role="Product Comparator & Reporter",
        goal="Compare products found across Store A, B, and C and recommend the best "
             "option with respect to what the user specifically asked about — never a "
             "different criterion presented as if it answers the question.",
        backstory="You are a meticulous shopping advisor. You never invent products or "
                   "prices that weren't given to you. If a store had no data, you say so "
                   "instead of guessing. Critically: search results can match the search "
                   "keywords without containing the actual attribute the user asked about "
                   "(e.g. a listing for a laptop with no battery-life spec, or a "
                   "replacement-battery accessory instead of the laptop itself). When that "
                   "happens, you say so plainly instead of quietly picking a winner on an "
                   "unrelated basis like price and presenting it as if it answered the "
                   "question. You always state explicitly, in the required 'Relevant data "
                   "found: yes/no' line, whether the requested attribute was actually "
                   "present before writing any recommendation. You write concisely — you "
                   "never pad your answer with numbered lists or multi-paragraph "
                   "justifications when a couple of sentences will do.",
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
            f"The user's question was: '{query}'.\n\n"
            "You MUST structure your output in exactly this order:\n\n"
            "1. The markdown comparison table (see expected_output for columns).\n"
            "2. A line starting with exactly 'Relevant data found: ' followed by 'yes' "
            "or 'no' — answering whether the Key Features text above actually contains "
            "information relevant to what was asked (e.g. for a battery life question, "
            "does any listing mention battery capacity, mAh, Wh, or hours of use — not "
            "just match the search keywords). A product matching the search terms is NOT "
            "the same as containing the requested data — e.g. a replacement-battery "
            "accessory listing, a listing for a different sub-model, or a laptop listing "
            "with no spec section for the attribute asked about does NOT count as 'yes'.\n"
            "3. If 'yes': one short paragraph (2-3 sentences, no lists) naming the single "
            "best pick on that basis and why.\n"
            "   If 'no': one short paragraph stating plainly that none of the results "
            "contain the requested information, and that price or other specs cannot "
            "substitute for it. Do NOT recommend a 'best' product on an unrelated basis "
            "in this case — just state clearly that the question can't be answered from "
            "these results.\n\n"
            "If a store's block above says it has no data, treat that store as having no "
            "data — do not invent products for it. "
            "IMPORTANT: every product link shown above must be preserved and shown in full "
            "in your output — never omit, shorten, or replace a link with just the product "
            "name."
        ),
        expected_output="A markdown comparison table with columns: Store, Product, Price, "
                         "Link, Key Features — where Link is the full clickable URL for each "
                         "product exactly as given above. Then a line reading exactly "
                         "'Relevant data found: yes' or 'Relevant data found: no'. Then ONE "
                         "short paragraph (2-3 sentences, no lists): if yes, the single best "
                         "pick, its link, and a brief reason why; if no, a clear statement "
                         "that the requested information wasn't found in these results "
                         "(no substitute recommendation).",
        agent=comparator_agent,
    )

    return Crew(
        agents=[comparator_agent],
        tasks=[compare_task],
        process=Process.sequential,
        verbose=True,
    )


def _sanity_check_relevance_claim(query: str, result_text: str) -> str:
    """
    Lightweight, non-LLM guard against the comparator claiming
    'Relevant data found: yes' without anything in its own output
    actually backing that up.

    This does NOT re-verify the model's reasoning (that's genuinely
    hard to guarantee against a small free-tier model) — it just does
    a crude keyword-overlap check between the query's content words
    and the full output text. If a 'yes' claim shares zero overlap
    with what was asked, that's a strong signal of a false positive
    (e.g. the model said yes but is actually recommending on price),
    so we append a visible warning rather than silently trusting the
    self-report. No overlap found -> warn; overlap found, or the
    claim is 'no' -> pass through unchanged.
    """
    match = re.search(r"Relevant data found:\s*(yes|no)", result_text, re.IGNORECASE)
    if not match or match.group(1).lower() != "yes":
        return result_text

    stopwords = {
        "which", "one", "has", "the", "best", "is", "a", "an", "for", "of", "in",
        "on", "with", "and", "or", "to", "does", "do", "are", "was", "were",
        "show", "me", "find", "what", "how",
    }
    query_terms = {
        w.lower().strip(".,?!\"'")
        for w in query.split()
        if w.lower() not in stopwords and len(w) > 2
    }

    text_lower = result_text.lower()
    if query_terms and not any(term in text_lower for term in query_terms):
        warning = (
            "\n\n⚠️ **Automated check**: the comparator marked this as "
            "'Relevant data found: yes', but none of the query terms "
            f"({', '.join(sorted(query_terms))}) appear anywhere in the output. "
            "This may be a false positive — verify manually before trusting this "
            "recommendation."
        )
        logger.warning(
            "Relevance sanity check failed for query %r — no term overlap found", query
        )
        return result_text + warning
    return result_text


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(
        "Groq rate limit hit (attempt %d/%d), retrying...",
        retry_state.attempt_number,
        5,
    ),
)
def run_product_search(query: str, history: list[dict] | None = None) -> str:
    """
    Run the three store searches directly in Python (no LLM turn, so
    no risk of an agent summarizing away real data), then let a single
    comparator agent reason over the guaranteed-verbatim results.

    history: prior turns from session_state["turns"], e.g.
        [{"query": "...", "answer": "...", "timestamp": "..."}, ...]
    in chronological order (most recent last). Optional — omit or pass
    None/[] for a fresh, context-free search.

    Before searching, the (possibly follow-up) query is resolved
    against the most recent prior turn via _resolve_query(), so a
    question like "which one has the best battery life" actually
    searches for the specific product it refers to instead of being
    searched literally.

    Placeholder/fake-looking store results are filtered out in Python
    before the comparator ever sees them (_filter_placeholder_results),
    and if every store ends up empty, a deterministic "no products
    found" message is returned without calling the LLM at all
    (_all_empty) — neither of these depend on the comparator noticing
    and reporting it correctly in prose.

    The comparator's output is required to include a literal
    "Relevant data found: yes/no" line before any recommendation, and
    a "yes" claim is spot-checked post-hoc for keyword overlap with
    the query (_sanity_check_relevance_claim) before being returned.
    """
    resolved_query = _resolve_query(query, history)
    search_results = _run_searches(resolved_query)
    search_results = _filter_placeholder_results(search_results)

    if _all_empty(search_results):
        logger.info("All stores returned empty results for query: %s", resolved_query)
        return (
            "No products were found for this search across any of the connected "
            "stores. Try rephrasing the query or broadening the search terms."
        )

    crew = build_crew(resolved_query, search_results)
    result = crew.kickoff()
    final = _strip_thinking(str(result))
    final = _sanity_check_relevance_claim(resolved_query, final)
    return final


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
        logger.warning(
            "Model response truncated mid-reasoning (no closing </think> tag); "
            "consider raising max_tokens in crew.py."
        )
        return (
            "The model's response was cut off while still reasoning and "
            "never produced a final answer. Try again, or increase "
            "max_tokens further in crew.py if this keeps happening."
        )

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()