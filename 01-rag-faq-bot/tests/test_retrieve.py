"""
test_retrieve.py — Lightweight smoke test for the retriever
============================================================
RAG FAQ Bot — Tests

The test runs an end-to-end ingestion + retrieval round-trip with a
small in-memory fixture so it does not depend on disk artifacts being
pre-built.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # noqa: E402

from src import ingest, retrieve   # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Fixture builder
# ──────────────────────────────────────────────────────────────────────
def _build_temp_index(tmp_path):
    """Re-point ingest constants to tmp_path and build a tiny index."""
    ingest.DATA_DIR = tmp_path / "data"
    ingest.ARTIFACTS_DIR = tmp_path / "artifacts"
    ingest.DOCS_PATH = ingest.DATA_DIR / "sample_faqs.txt"
    ingest.INDEX_PATH = ingest.ARTIFACTS_DIR / "faqs.faiss"
    ingest.META_PATH = ingest.ARTIFACTS_DIR / "chunks.json"

    retrieve.ARTIFACTS_DIR = ingest.ARTIFACTS_DIR
    retrieve.INDEX_PATH = ingest.INDEX_PATH
    retrieve.META_PATH = ingest.META_PATH

    ingest.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ingest.DOCS_PATH.write_text(
        "### What is FAISS?\nFAISS is a vector search library.\n\n"
        "### What is RAG?\nRAG retrieves context for an LLM.\n",
        encoding="utf-8",
    )
    ingest.main()


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
def test_retrieves_relevant_chunk(tmp_path):
    _build_temp_index(tmp_path)
    hits = retrieve.search("vector search library", top_k=2)
    assert hits, "expected at least one hit"
    assert "FAISS" in hits[0].text


def test_score_floor_filters_irrelevant(tmp_path):
    _build_temp_index(tmp_path)
    # Push the floor very high to force a refusal-equivalent empty list
    retrieve.MIN_SIM = 0.99
    hits = retrieve.search("totally unrelated cooking recipe", top_k=2)
    assert hits == []
