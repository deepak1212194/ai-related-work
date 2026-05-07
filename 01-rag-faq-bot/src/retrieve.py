"""
retrieve.py — Semantic Search & Classification
================================================
Semantic Search & Classification Service — Module 2

Given a natural-language query, retrieves the most relevant documents
from the FAISS index and classifies the query into a domain/category
based on the retrieved evidence.

Classification strategy:
  1. Retrieve top-K neighbours from the index.
  2. Aggregate category votes from retrieved documents (weighted by score).
  3. Return the dominant category with a confidence measure.

This mirrors production patterns where:
  - Job descriptions are classified into domains via nearest-neighbour
    voting over an embedding index.
  - Talent profiles are matched to job categories using weighted retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedDoc:
    """A single retrieved document with its similarity score."""
    text: str
    score: float
    source: str = ""
    category: str = ""
    chunk_id: int = 0


@dataclass
class ClassificationResult:
    """Aggregated classification from retrieved evidence."""
    predicted_category: str
    confidence: float                      # 0–1, weighted vote share
    category_scores: Dict[str, float] = field(default_factory=dict)
    evidence_count: int = 0


@dataclass
class SearchResult:
    """Full search response: retrieved docs + optional classification."""
    query: str
    documents: List[RetrievedDoc]
    classification: Optional[ClassificationResult] = None
    elapsed_ms: int = 0


# ──────────────────────────────────────────────────────────────────────
# Module 1: Retrieval
# ──────────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    model,
    index,
    metadata: List[Dict],
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[RetrievedDoc]:
    """
    Embed the query and search the FAISS index.

    Returns up to top_k documents with score >= min_score.
    """
    qvec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, idxs = index.search(qvec, top_k)
    scores, idxs = scores[0], idxs[0]

    results: List[RetrievedDoc] = []
    for score, idx in zip(scores, idxs):
        if idx == -1 or score < min_score:
            continue
        doc = metadata[idx]
        results.append(RetrievedDoc(
            text=doc["text"],
            score=float(score),
            source=doc.get("source", ""),
            category=doc.get("category", ""),
            chunk_id=doc.get("chunk_id", 0),
        ))
    return results


# ──────────────────────────────────────────────────────────────────────
# Module 2: Classification by weighted vote
# ──────────────────────────────────────────────────────────────────────
def classify_from_results(
    docs: List[RetrievedDoc],
) -> Optional[ClassificationResult]:
    """
    Classify the query by aggregating category votes from retrieved docs.

    Each document's vote is weighted by its similarity score.  The
    category with the highest total weight wins.

    Returns None if no documents have categories.
    """
    # Filter docs that have a category label
    labelled = [d for d in docs if d.category]
    if not labelled:
        return None

    # Weighted vote aggregation
    votes: Dict[str, float] = {}
    for d in labelled:
        votes[d.category] = votes.get(d.category, 0.0) + d.score

    total_weight = sum(votes.values())
    if total_weight == 0:
        return None

    # Sort by weight descending
    sorted_cats = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_weight = sorted_cats[0]

    return ClassificationResult(
        predicted_category=best_cat,
        confidence=round(best_weight / total_weight, 4),
        category_scores={
            cat: round(w / total_weight, 4) for cat, w in sorted_cats
        },
        evidence_count=len(labelled),
    )


# ──────────────────────────────────────────────────────────────────────
# Module 3: Evaluation metrics
# ──────────────────────────────────────────────────────────────────────
def evaluate_retrieval(
    queries: List[str],
    ground_truth_categories: List[str],
    model,
    index,
    metadata: List[Dict],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate retrieval quality with standard IR metrics.

    Metrics:
      - accuracy: fraction of queries where top-1 category matches ground truth
      - mrr: Mean Reciprocal Rank of the first correct category
      - recall_at_k: fraction of queries where correct category appears in top-K
    """
    correct_top1 = 0
    reciprocal_ranks = []
    recall_hits = 0

    for query, true_cat in zip(queries, ground_truth_categories):
        docs = retrieve(query, model, index, metadata, top_k=top_k)
        categories = [d.category for d in docs if d.category]

        # Top-1 accuracy
        if categories and categories[0] == true_cat:
            correct_top1 += 1

        # MRR
        found = False
        for rank, cat in enumerate(categories, 1):
            if cat == true_cat:
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        if not found:
            reciprocal_ranks.append(0.0)

        # Recall@K
        if true_cat in categories:
            recall_hits += 1

    n = len(queries)
    return {
        "accuracy": round(correct_top1 / n, 4) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
        "recall_at_k": round(recall_hits / n, 4) if n else 0.0,
        "k": top_k,
        "n_queries": n,
    }
