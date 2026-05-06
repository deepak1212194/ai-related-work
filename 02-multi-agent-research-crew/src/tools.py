"""
tools.py — Tool Functions
=========================
Multi-Agent Research Crew — Module 2

Tools the researcher agent can call. The web search is a deterministic
stub by default so the demo runs fully offline. Replace with a real
search provider for live use (Tavily, SerpAPI, Brave, etc.).
"""

from typing import Callable


# ──────────────────────────────────────────────────────────────────────
# Module 1: Web search stub (deterministic, offline)
# ──────────────────────────────────────────────────────────────────────
_STUB_RESULTS = {
    "moe": (
        "Mixture-of-Experts LLMs activate only a subset of parameters per "
        "token (e.g., Qwen3.5-35B-A3B activates ~3B per token), giving lower "
        "inference cost than a dense model of the same total size, at the "
        "expense of more memory at rest and harder fine-tuning."
    ),
    "rag": (
        "Retrieval-Augmented Generation grounds an LLM's response in "
        "retrieved documents, reducing hallucination but adding retrieval "
        "latency and a dependency on the retrieval corpus quality."
    ),
    "agent": (
        "Agentic systems trade single-call latency for compositional "
        "reasoning. A well-defined crew of 3-5 specialists generally "
        "outperforms a single self-prompting model on multi-step tasks."
    ),
    "default": (
        "No high-quality search snippet available for this query. Researcher "
        "should answer from parametric knowledge and flag the gap."
    ),
}


def web_search_stub(query: str) -> str:
    """Return a canned snippet for the closest-matching keyword."""
    q = query.lower()
    for k, v in _STUB_RESULTS.items():
        if k in q:
            return v
    return _STUB_RESULTS["default"]


# ──────────────────────────────────────────────────────────────────────
# Module 2: Calculator
# ──────────────────────────────────────────────────────────────────────
def calculator(expr: str) -> str:
    """
    Evaluate a small numeric expression. Restricted to digits, basic
    operators and parentheses — no `eval` on arbitrary input.
    """
    allowed = set("0123456789+-*/(). ")
    if not expr or any(ch not in allowed for ch in expr):
        return "ERR: only digits and + - * / ( ) . are allowed"
    try:
        # Safe-ish: we already restricted the alphabet above
        return str(eval(expr, {"__builtins__": {}}, {}))   # noqa: S307
    except Exception as e:                                  # noqa: BLE001
        return f"ERR: {e}"


# ──────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────
TOOLS: dict[str, Callable[[str], str]] = {
    "web_search_stub": web_search_stub,
    "calculator": calculator,
}
