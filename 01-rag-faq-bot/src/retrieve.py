"""
retrieve.py — Top-K Semantic Retrieval with Score Floor
========================================================
RAG FAQ Bot — Module 2

Loads the FAISS index produced by ingest.py and answers a query with the
top-K most similar chunks plus their cosine-similarity scores. Applies a
score floor so the downstream generator can refuse on weak retrieval.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

INDEX_PATH = ARTIFACTS_DIR / "faqs.faiss"
META_PATH = ARTIFACTS_DIR / "chunks.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MIN_SIM = 0.45    # cosine-sim floor; below this we return no results


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    text: str
    score: float


# ──────────────────────────────────────────────────────────────────────
# Module 1: Load Artifacts
# ──────────────────────────────────────────────────────────────────────
def load_artifacts() -> tuple[faiss.Index, List[str], SentenceTransformer]:
    """Load FAISS index, chunk metadata, and the embedding model."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run `python -m src.ingest` first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, meta["chunks"], model


# ──────────────────────────────────────────────────────────────────────
# Module 2: Search
# ──────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """
    Embed the query and return the top-K matching chunks above MIN_SIM.

    Returns an empty list when no chunk crosses the floor — this is the
    signal the generator uses to refuse instead of hallucinate.
    """
    index, chunks, model = load_artifacts()

    qvec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, idxs = index.search(qvec, top_k)
    scores, idxs = scores[0], idxs[0]

    results: List[RetrievedChunk] = []
    for s, i in zip(scores, idxs):
        if i == -1:
            continue
        if s < MIN_SIM:
            continue
        results.append(RetrievedChunk(text=chunks[i], score=float(s)))

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI helper (for ad-hoc inspection)
# ──────────────────────────────────────────────────────────────────────
def _format(results: List[RetrievedChunk]) -> str:
    if not results:
        return "(no chunks above similarity floor)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] score={r.score:.3f}")
        lines.append(f"      {r.text[:120]}{'...' if len(r.text) > 120 else ''}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Top-K retrieval over the FAQ index.")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K)
    args = parser.parse_args()

    hits = search(args.question, top_k=args.top_k)
    print(f"[RETRIEVE] Query: {args.question}")
    print(_format(hits))
