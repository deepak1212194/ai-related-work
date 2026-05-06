"""
deps.py — Process-wide CLIP model + index cache
================================================
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class AppState:
    model: CLIPModel | None = None
    processor: CLIPProcessor | None = None
    image_vecs: np.ndarray | None = None
    image_paths: list[str] = field(default_factory=list)

    @property
    def index_built(self) -> bool:
        return (self.image_vecs is not None
                and self.image_vecs.size > 0
                and len(self.image_paths) == self.image_vecs.shape[0])


state = AppState()

VEC_PATH = settings.artifacts_dir / "image_vectors.npz"
MANIFEST_PATH = settings.artifacts_dir / "manifest.json"


# ──────────────────────────────────────────────────────────────────────
# Module 1: Loaders
# ──────────────────────────────────────────────────────────────────────
def load_model() -> None:
    if state.model is not None:
        return
    log.info("[DEPS] Loading CLIP: %s", settings.model_name)
    state.model = CLIPModel.from_pretrained(settings.model_name).eval()
    state.processor = CLIPProcessor.from_pretrained(settings.model_name)


def load_index() -> None:
    if not VEC_PATH.exists() or not MANIFEST_PATH.exists():
        log.warning("[DEPS] No image index on disk yet.")
        state.image_vecs = None
        state.image_paths = []
        return
    state.image_vecs = np.load(VEC_PATH)["vecs"].astype("float32")
    state.image_paths = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )["paths"]
    log.info("[DEPS] Loaded index: %d images", len(state.image_paths))


def warmup() -> None:
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    load_model()
    load_index()


# ──────────────────────────────────────────────────────────────────────
# Module 2: Online query encoder
# ──────────────────────────────────────────────────────────────────────
def encode_text(query: str) -> np.ndarray:
    assert state.model is not None and state.processor is not None
    with torch.no_grad():
        inputs = state.processor(text=[query], return_tensors="pt", padding=True)
        feats = state.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")
