"""
generate.py — Constrained-Context LLM Answering
================================================
RAG FAQ Bot — Module 3

Takes a user question + retrieved chunks and produces a structured answer.
The system prompt instructs the model to answer ONLY from the provided
context. The output is validated against a Pydantic schema so calling
code can detect refusals deterministically.

Falls back to a deterministic "echo" generator when OPENAI_API_KEY is
not set — useful for offline demos and CI.
"""

import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .retrieve import RetrievedChunk

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS = 400
TEMPERATURE = 0.0


# ──────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────
class Answer(BaseModel):
    """Structured response from the FAQ bot."""

    status: Literal["answered", "refused"] = Field(
        description="`refused` means no chunk crossed the similarity floor."
    )
    answer: str = Field(description="Final answer text, or refusal reason.")
    citations: List[str] = Field(
        default_factory=list,
        description="Snippets of the retrieved chunks used to ground the answer.",
    )


# ──────────────────────────────────────────────────────────────────────
# Module 1: Prompt construction
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a factual FAQ assistant. Answer ONLY using the provided "
    "context. If the context does not directly support an answer, reply "
    "with the exact string `IDK`. Do not use outside knowledge."
)


def build_user_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """Assemble the user-message body with context blocks."""
    context_blocks = "\n\n".join(
        f"[Chunk {i+1} | score={c.score:.3f}]\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


# ──────────────────────────────────────────────────────────────────────
# Module 2: LLM call (with offline fallback)
# ──────────────────────────────────────────────────────────────────────
def _call_openai(system: str, user: str) -> Optional[str]:
    """Call OpenAI when a key is present; return None otherwise."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    from openai import OpenAI
    client = OpenAI()

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def _offline_stub(chunks: List[RetrievedChunk]) -> str:
    """Deterministic offline answer — concatenates the top chunks."""
    if not chunks:
        return "IDK"
    return "OFFLINE-STUB ▸ " + chunks[0].text.split("\n", 1)[-1].strip()


# ──────────────────────────────────────────────────────────────────────
# Module 3: Public API
# ──────────────────────────────────────────────────────────────────────
def generate(question: str, chunks: List[RetrievedChunk]) -> Answer:
    """
    Produce a structured Answer for `question` given retrieved `chunks`.

    Refuses immediately when `chunks` is empty (the retriever's score
    floor already failed) — saving an LLM call.
    """
    if not chunks:
        return Answer(
            status="refused",
            answer="No relevant context found above the similarity threshold.",
        )

    user = build_user_prompt(question, chunks)
    raw = _call_openai(SYSTEM_PROMPT, user) or _offline_stub(chunks)

    if raw.strip().upper() == "IDK":
        return Answer(
            status="refused",
            answer="The retrieved context did not directly support an answer.",
            citations=[c.text[:200] for c in chunks],
        )

    return Answer(
        status="answered",
        answer=raw,
        citations=[c.text[:200] for c in chunks],
    )
