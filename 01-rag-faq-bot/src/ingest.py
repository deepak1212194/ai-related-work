"""
ingest.py — Document Ingestion & FAISS Index Build
===================================================
Semantic Search & Classification Service — Module 1

Reads document collections (text, CSV, JSON), chunks them by semantic
boundaries, embeds chunks with a sentence transformer, and persists
a FAISS index plus metadata.  Supports incremental re-indexing.

This mirrors production patterns for:
  - Job-description domain classification
  - Talent-profile semantic indexing
  - Document retrieval over constrained taxonomies
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
META_PATH = ARTIFACTS_DIR / "metadata.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ──────────────────────────────────────────────────────────────────────
# Document loaders
# ──────────────────────────────────────────────────────────────────────
def _load_txt(path: Path) -> List[Dict]:
    """Load a heading-delimited text file into chunks."""
    raw = path.read_text(encoding="utf-8")
    delimiter = "###"
    sections = [s.strip() for s in raw.split(delimiter) if s.strip()]
    return [
        {"text": s, "source": path.name, "chunk_id": i}
        for i, s in enumerate(sections)
    ]


def _load_csv(path: Path) -> List[Dict]:
    """Load a CSV where each row is a document (uses 'text' or first column)."""
    import csv
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row.get("text") or row.get("description") or list(row.values())[0]
            category = row.get("category") or row.get("label") or row.get("domain", "")
            docs.append({
                "text": str(text).strip(),
                "source": path.name,
                "chunk_id": i,
                "category": str(category).strip(),
            })
    return docs


def _load_json(path: Path) -> List[Dict]:
    """Load a JSON array of documents."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("documents", data.get("items", [data]))
    docs = []
    for i, item in enumerate(data):
        text = item.get("text") or item.get("description") or str(item)
        docs.append({
            "text": str(text).strip(),
            "source": path.name,
            "chunk_id": i,
            "category": item.get("category", ""),
        })
    return docs


LOADERS = {".txt": _load_txt, ".csv": _load_csv, ".json": _load_json}


# ──────────────────────────────────────────────────────────────────────
# Module 1: Read & Chunk
# ──────────────────────────────────────────────────────────────────────
def load_documents(data_dir: Path) -> List[Dict]:
    """
    Scan data_dir for supported files and load all documents.

    Returns a list of dicts with at least {text, source, chunk_id}.
    """
    all_docs: List[Dict] = []
    for ext, loader in LOADERS.items():
        for path in sorted(data_dir.glob(f"*{ext}")):
            try:
                docs = loader(path)
                all_docs.extend(docs)
                print(f"[INGEST] {path.name}: {len(docs)} chunks")
            except Exception as e:
                print(f"[INGEST] WARN: skipping {path.name}: {e}")
    if not all_docs:
        print(f"[INGEST] No documents found in {data_dir}")
    return all_docs


# ──────────────────────────────────────────────────────────────────────
# Module 2: Embed
# ──────────────────────────────────────────────────────────────────────
def embed_documents(
    docs: List[Dict],
    model_name: str = EMBED_MODEL_NAME,
) -> np.ndarray:
    """Encode document texts into a (N, D) float32 array, L2-normalised."""
    print(f"[EMBED] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]
    print(f"[EMBED] Encoding {len(texts)} documents...")
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"[EMBED] Output shape: {vecs.shape}")
    return vecs


# ──────────────────────────────────────────────────────────────────────
# Module 3: Build FAISS Index
# ──────────────────────────────────────────────────────────────────────
def build_index(vecs: np.ndarray) -> faiss.Index:
    """
    Build a flat inner-product index.

    For under ~1M vectors, exact search is fast and avoids accuracy loss
    from approximate methods.  Switch to IVF or HNSW above that scale.
    """
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"[INDEX] Built flat-IP index: dim={dim}, ntotal={index.ntotal}")
    return index


# ──────────────────────────────────────────────────────────────────────
# Module 4: Persist
# ──────────────────────────────────────────────────────────────────────
def save_artifacts(
    index: faiss.Index,
    docs: List[Dict],
    vecs: np.ndarray,
) -> None:
    """Persist the FAISS index and document metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[SAVE] FAISS index → {INDEX_PATH}")

    # Build metadata: store text, source, category, and a content hash
    meta = {
        "model": EMBED_MODEL_NAME,
        "dim": int(vecs.shape[1]),
        "n_docs": len(docs),
        "documents": [
            {
                "text": d["text"],
                "source": d.get("source", ""),
                "category": d.get("category", ""),
                "chunk_id": d.get("chunk_id", i),
                "hash": hashlib.md5(d["text"].encode()).hexdigest()[:12],
            }
            for i, d in enumerate(docs)
        ],
    }
    META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[SAVE] Metadata → {META_PATH}")


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main(data_dir: Optional[Path] = None) -> int:
    """Run the full ingestion pipeline.  Returns document count."""
    src = data_dir or DATA_DIR
    if not src.exists():
        sys.exit(f"[ERR] Data directory not found: {src}")

    docs = load_documents(src)
    if not docs:
        sys.exit("[ERR] No documents loaded — check data/ directory")

    vecs = embed_documents(docs)
    index = build_index(vecs)
    save_artifacts(index, docs, vecs)

    print(f"[DONE] Ingested {len(docs)} documents.")
    return len(docs)


if __name__ == "__main__":
    main()
