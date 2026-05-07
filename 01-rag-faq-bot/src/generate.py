"""
generate.py — Answer Generation (optional LLM layer)
=====================================================
Semantic Search & Classification Service — Module 3

When an LLM is available (OPENAI_API_KEY set), generates a synthesised
answer grounded in retrieved documents.  When no LLM is available, the
service still works fully: it returns retrieved documents and
classification results — the LLM layer is additive, not required.

Hallucination guard: the prompt explicitly constrains the LLM to only
use information from the provided context passages.
"""

import os
from dataclasses import dataclass
from typing import List

from .retrieve import RetrievedDoc


@dataclass
class GeneratedAnswer:
    """LLM-generated answer with grounding status."""
    answer: str
    status: str       # "grounded" | "no_context" | "llm_unavailable"
    model: str = ""


# ──────────────────────────────────────────────────────────────────────
# System prompt — hallucination guard
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI assistant that answers questions ONLY using the "
    "provided context passages. Rules:\n"
    "1. Base your answer entirely on the context below.\n"
    "2. If the context does not contain enough information, say "
    "\"I don't have enough information to answer this.\"\n"
    "3. Never invent facts or use knowledge outside the context.\n"
    "4. Cite the passage number(s) you used, e.g. [1], [2].\n"
    "5. Keep answers concise and direct."
)


def _build_context(docs: List[RetrievedDoc]) -> str:
    """Format retrieved documents as numbered passages."""
    parts = []
    for i, d in enumerate(docs, 1):
        cat_label = f" [{d.category}]" if d.category else ""
        parts.append(f"[{i}]{cat_label} (score: {d.score:.3f})\n{d.text}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────────────────────────────
def generate(
    question: str,
    docs: List[RetrievedDoc],
) -> GeneratedAnswer:
    """
    Generate a grounded answer using retrieved documents.

    Falls back gracefully when:
      - No documents retrieved → returns "no_context"
      - No API key set → returns "llm_unavailable"
      - API error → returns error message with "llm_unavailable"
    """
    if not docs:
        return GeneratedAnswer(
            answer="No relevant documents found for this query.",
            status="no_context",
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # Graceful degradation: return a summary without LLM
        top = docs[0]
        summary = (
            f"Top match (score: {top.score:.3f}): {top.text[:300]}"
        )
        if top.category:
            summary = f"Category: {top.category}\n\n{summary}"
        return GeneratedAnswer(
            answer=summary,
            status="llm_unavailable",
        )

    # Call LLM with grounding context
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        context = _build_context(docs)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()
        return GeneratedAnswer(
            answer=answer,
            status="grounded",
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )
    except Exception as e:
        return GeneratedAnswer(
            answer=f"LLM call failed: {e}. Returning top retrieved document.",
            status="llm_unavailable",
        )
