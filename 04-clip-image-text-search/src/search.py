"""
search.py — Text-Query Image Search
====================================
CLIP Image-Text Search — Module 2

Given a natural-language query, encode it with CLIP's text tower and
return the top-K cached images by cosine similarity.

Usage:
    python -m src.search --query "a photo of a sunset" --top-k 3
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

VEC_PATH = ARTIFACTS_DIR / "image_vectors.npz"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_TOP_K = 3


# ──────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Match:
    path: str
    score: float


# ──────────────────────────────────────────────────────────────────────
# Module 1: Load cache
# ──────────────────────────────────────────────────────────────────────
def load_cache() -> tuple[np.ndarray, list[str]]:
    """Load image vectors + manifest produced by embed_images.py."""
    if not VEC_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Cache missing. Run `python -m src.embed_images` first."
        )

    vecs = np.load(VEC_PATH)["vecs"].astype("float32")
    paths = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["paths"]
    print(f"[LOAD] Vectors {vecs.shape}  |  {len(paths)} images")
    return vecs, paths


# ──────────────────────────────────────────────────────────────────────
# Module 2: Encode query
# ──────────────────────────────────────────────────────────────────────
def encode_query(query: str) -> np.ndarray:
    """Encode the text query with CLIP's text tower; return L2-normed vector."""
    model = CLIPModel.from_pretrained(MODEL_NAME).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy().astype("float32")


# ──────────────────────────────────────────────────────────────────────
# Module 3: Search
# ──────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[Match]:
    """Return the top-K image matches for `query`."""
    image_vecs, paths = load_cache()
    qvec = encode_query(query)

    sims = (image_vecs @ qvec.T).squeeze()       # cosine because both L2-normed
    top_idx = np.argsort(-sims)[:top_k]

    return [Match(path=paths[i], score=float(sims[i])) for i in top_idx]


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP image-text search.")
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    matches = search(args.query, top_k=args.top_k)

    print(f"\n[SEARCH] Query: {args.query}")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. score={m.score:.3f}  {m.path}")


if __name__ == "__main__":
    main()
