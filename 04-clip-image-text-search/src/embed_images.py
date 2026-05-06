"""
embed_images.py — Image Embedding & Cache
==========================================
CLIP Image-Text Search — Module 1

Loads CLIP, encodes every image in `data/images/` into a single (N, D)
matrix, and caches the matrix plus a JSON manifest of file paths for the
search step.

This script runs end-to-end: embed_images.py → search.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

VEC_PATH = ARTIFACTS_DIR / "image_vectors.npz"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 8


# ──────────────────────────────────────────────────────────────────────
# Module 1: List images
# ──────────────────────────────────────────────────────────────────────
def list_images(folder: Path) -> list[Path]:
    """Return all image files in `folder` (non-recursive)."""
    if not folder.exists():
        sys.exit(f"[ERR] Image folder not found: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        sys.exit(f"[ERR] No images found in {folder}")
    print(f"[SCAN] Found {len(paths)} images in {folder}")
    return paths


# ──────────────────────────────────────────────────────────────────────
# Module 2: Encode
# ──────────────────────────────────────────────────────────────────────
def encode_images(paths: list[Path]) -> np.ndarray:
    """Run CLIP image encoder over `paths` in batches; return L2-normed vectors."""
    print(f"[MODEL] Loading {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    all_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = processor(images=images, return_tensors="pt")
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise

            all_vecs.append(feats.cpu().numpy().astype("float32"))
            print(f"[ENCODE] {start + len(batch_paths):4d}/{len(paths)}")

    return np.vstack(all_vecs)


# ──────────────────────────────────────────────────────────────────────
# Module 3: Persist
# ──────────────────────────────────────────────────────────────────────
def save_cache(vecs: np.ndarray, paths: list[Path]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(VEC_PATH, vecs=vecs)
    print(f"[SAVE] Vectors → {VEC_PATH}  shape={vecs.shape}")

    manifest = {"paths": [str(p.relative_to(BASE_DIR)) for p in paths]}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SAVE] Manifest → {MANIFEST_PATH}  (n={len(paths)})")


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    paths = list_images(IMAGES_DIR)
    vecs = encode_images(paths)
    save_cache(vecs, paths)
    print("[DONE] Image embedding cache built.")


if __name__ == "__main__":
    main()
