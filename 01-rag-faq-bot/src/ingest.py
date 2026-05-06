"""
ingest.py — Document Ingestion & Index Build
============================================
RAG FAQ Bot — Module 1

Reads FAQ-style documents, chunks them on heading markers, embeds the
chunks with a sentence transformer, and persists a FAISS index plus the
chunk metadata to disk.

This script runs end-to-end: ingest.py → retrieve.py → generate.py
"""

import json
import sys
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DOCS_PATH = DATA_DIR / "sample_faqs.txt"
INDEX_PATH = ARTIFACTS_DIR / "faqs.faiss"
META_PATH = ARTIFACTS_DIR / "chunks.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_DELIMITER = "###"   # heading marker used in sample_faqs.txt


# ──────────────────────────────────────────────────────────────────────
# Module 1: Read & Chunk
# ──────────────────────────────────────────────────────────────────────
def read_chunks(docs_path: Path) -> List[str]:
    """
    Split the source file into chunks at the heading delimiter.

    WHY chunking on headings (not on N tokens):
    ────────────────────────────────────────────
    For FAQ-style data, each Q-A pair is a self-contained unit. Splitting
    on a fixed token count would slice questions away from their answers,
    which destroys retrieval quality. Heading-based chunking preserves the
    semantic boundary the author already encoded.
    """
    print(f"[INGEST] Reading {docs_path}")
    raw = docs_path.read_text(encoding="utf-8")

    chunks = [
        c.strip()
        for c in raw.split(CHUNK_DELIMITER)
        if c.strip()
    ]
    print(f"[INGEST] Found {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────────────────────────────
# Module 2: Embed
# ──────────────────────────────────────────────────────────────────────
def embed_chunks(chunks: List[str], model_name: str) -> np.ndarray:
    """Encode chunks into a (N, D) float32 array, L2-normalised."""
    print(f"[EMBED] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[EMBED] Encoding {len(chunks)} chunks...")
    vecs = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so we can use inner-product as cosine
    ).astype("float32")

    print(f"[EMBED] Output shape: {vecs.shape}")
    return vecs


# ──────────────────────────────────────────────────────────────────────
# Module 3: Build FAISS Index
# ──────────────────────────────────────────────────────────────────────
def build_index(vecs: np.ndarray) -> faiss.Index:
    """
    Build a flat inner-product index.

    For under ~1M vectors, exact search on CPU is fast and saves the
    accuracy hit of approximate methods. Switch to IVF or HNSW above
    that scale.
    """
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # IP on normalised vectors == cosine
    index.add(vecs)
    print(f"[INDEX] Built flat-IP index: dim={dim}, ntotal={index.ntotal}")
    return index


# ──────────────────────────────────────────────────────────────────────
# Module 4: Persist
# ──────────────────────────────────────────────────────────────────────
def save_artifacts(index: faiss.Index, chunks: List[str]) -> None:
    """Persist the index and chunk metadata for the retriever."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[SAVE] FAISS index → {INDEX_PATH}")

    META_PATH.write_text(
        json.dumps({"chunks": chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[SAVE] Chunk metadata → {META_PATH}")


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not DOCS_PATH.exists():
        sys.exit(f"[ERR] Source docs not found: {DOCS_PATH}")

    chunks = read_chunks(DOCS_PATH)
    vecs = embed_chunks(chunks, EMBED_MODEL_NAME)
    index = build_index(vecs)
    save_artifacts(index, chunks)

    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    main()
