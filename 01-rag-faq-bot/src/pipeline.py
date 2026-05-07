"""
pipeline.py — CLI entry point for search and evaluation
=========================================================
Usage:
    python -m src.pipeline -q "machine learning engineer"
    python -m src.pipeline --eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from .ingest import ARTIFACTS_DIR, INDEX_PATH, META_PATH, EMBED_MODEL_NAME
from .retrieve import retrieve, classify_from_results, evaluate_retrieval
from .generate import generate


def _load_assets():
    """Load the model, index, and metadata."""
    if not INDEX_PATH.exists():
        sys.exit("[ERR] Index not found. Run: python -m src.ingest")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, index, meta["documents"]


def search(query: str, top_k: int = 5):
    """Run a search query and print results."""
    model, index, docs = _load_assets()

    t0 = time.perf_counter()
    results = retrieve(query, model, index, docs, top_k=top_k)
    classification = classify_from_results(results)
    answer = generate(query, results)
    elapsed = int((time.perf_counter() - t0) * 1000)

    print(f"\n{'='*60}")
    print(f"  Query: {query}")
    print(f"  Results: {len(results)} | Elapsed: {elapsed}ms")
    print(f"{'='*60}\n")

    if classification:
        print(f"  Classification: {classification.predicted_category} "
              f"(confidence: {classification.confidence:.1%})")
        print(f"  Category scores: {classification.category_scores}\n")

    for i, r in enumerate(results, 1):
        cat = f" [{r.category}]" if r.category else ""
        print(f"  [{i}] score={r.score:.3f}{cat}")
        print(f"      {r.text[:200]}...\n")

    print(f"  Answer ({answer.status}):")
    print(f"  {answer.answer}\n")


def evaluate():
    """Run evaluation on labelled data."""
    model, index, docs = _load_assets()

    # Build eval set from indexed docs that have categories
    labelled = [d for d in docs if d.get("category")]
    if len(labelled) < 3:
        print("[EVAL] Not enough labelled documents for evaluation.")
        return

    queries = [d["text"][:100] for d in labelled]
    categories = [d["category"] for d in labelled]

    metrics = evaluate_retrieval(queries, categories, model, index, docs)
    print(f"\n{'='*60}")
    print("  Retrieval Evaluation")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("-q", "--query", type=str, help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    elif args.query:
        search(args.query, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
