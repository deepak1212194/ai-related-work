"""
deps.py — Shared singletons (model, FAISS index)
=================================================
RAG FAQ Service — App layer

Loading the embedding model takes 1–3 seconds; loading the FAISS index
takes another second. Doing this on every request would kill p99
latency, so we cache them as a single global state object that the
FastAPI startup hook populates once.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from .config import settings

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Application state
# ──────────────────────────────────────────────────────────────────────
@dataclass
class AppState:
    """Process-wide singletons. Mutated only at startup / on /api/ingest."""

    model: SentenceTransformer | None = None
    index: faiss.Index | None = None
    chunks: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.index is not None


state = AppState()


# ──────────────────────────────────────────────────────────────────────
# Module 1: Loaders
# ──────────────────────────────────────────────────────────────────────
def load_model() -> None:
    """Load the embedding model into the global state (idempotent)."""
    if state.model is not None:
        return
    log.info("[DEPS] Loading embedding model: %s", settings.embed_model)
    state.model = SentenceTransformer(settings.embed_model)


def load_index() -> None:
    """Load the FAISS index + chunk metadata if both exist on disk."""
    idx_path: Path = settings.index_path
    meta_path: Path = settings.meta_path

    if not idx_path.exists() or not meta_path.exists():
        log.warning(
            "[DEPS] Index artifacts missing — run POST /api/ingest first"
            " (looked at %s, %s)", idx_path, meta_path,
        )
        state.index = None
        state.chunks = []
        return

    state.index = faiss.read_index(str(idx_path))
    state.chunks = json.loads(meta_path.read_text(encoding="utf-8"))["chunks"]
    log.info("[DEPS] Loaded index: ntotal=%d  chunks=%d",
             state.index.ntotal, len(state.chunks))


# ──────────────────────────────────────────────────────────────────────
# Module 2: FastAPI startup hook
# ──────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """Called from main.py on app startup."""
    load_model()
    load_index()
