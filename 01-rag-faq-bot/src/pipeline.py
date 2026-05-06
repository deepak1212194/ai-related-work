"""
pipeline.py — End-to-End RAG Pipeline
======================================
RAG FAQ Bot — Module 4

CLI entry that wires retrieve.py and generate.py into a single call.

Usage:
    python -m src.pipeline --question "What is FAISS?"
"""

import argparse
import json

from .generate import Answer, generate
from .retrieve import search


# ──────────────────────────────────────────────────────────────────────
# Module 1: One-shot run
# ──────────────────────────────────────────────────────────────────────
def answer_question(question: str, top_k: int = 3) -> Answer:
    """Retrieve top-K chunks and ask the generator for a structured answer."""
    print(f"[RAG] Question: {question}")
    chunks = search(question, top_k=top_k)
    print(f"[RAG] Retrieved {len(chunks)} chunks above similarity floor")
    return generate(question, chunks)


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline once.")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--top-k", "-k", type=int, default=3)
    args = parser.parse_args()

    result = answer_question(args.question, top_k=args.top_k)

    print("\n[RAG] Result:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
