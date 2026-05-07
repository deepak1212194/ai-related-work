"""
main.py — FastAPI application
==============================
Semantic Search & Classification Service

Endpoints:
  GET  /health          → service status
  POST /api/ingest      → rebuild the index from data/
  POST /api/search      → search + classify a query
  GET  /api/stats       → index statistics
  POST /api/evaluate    → run retrieval evaluation
  GET  /                → browser UI
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sentence_transformers import SentenceTransformer

from .config import settings, UI_DIR
from .schemas import (
    SearchRequest,
    SearchResponse,
    SearchHit,
    ClassificationInfo,
    IngestResponse,
    StatsResponse,
    EvalResponse,
    HealthResponse,
)
from src import ingest as ingest_mod
from src.retrieve import retrieve, classify_from_results, evaluate_retrieval
from src.generate import generate

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Singleton state
# ──────────────────────────────────────────────────────────────────────
class _State:
    model: Optional[SentenceTransformer] = None
    index: Optional[faiss.Index] = None
    metadata: List[dict] = []
    is_ready: bool = False

state = _State()


def _load_index():
    """Load or reload the FAISS index and metadata."""
    idx_path = ingest_mod.INDEX_PATH
    meta_path = ingest_mod.META_PATH
    if idx_path.exists() and meta_path.exists():
        state.index = faiss.read_index(str(idx_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        state.metadata = raw.get("documents", [])
        log.info("[LOAD] Index: %d docs", state.index.ntotal)
    else:
        log.warning("[LOAD] No index found — POST /api/ingest to build one")


def _warmup():
    """Load model and index at startup."""
    log.info("[STARTUP] Loading embedding model...")
    state.model = SentenceTransformer(ingest_mod.EMBED_MODEL_NAME)
    _load_index()
    state.is_ready = True
    log.info("[STARTUP] Ready")


# ──────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="Semantic Search & Classification Service",
    description=(
        "Embedding-based document search with weighted-vote classification. "
        "Mirrors production patterns for domain mapping and talent matching."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if state.is_ready else "degraded",
        index_loaded=state.index is not None,
        embed_model=ingest_mod.EMBED_MODEL_NAME,
        documents_indexed=len(state.metadata),
    )


# ──────────────────────────────────────────────────────────────────────
# POST /api/ingest
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/ingest", response_model=IngestResponse, tags=["index"])
def ingest_endpoint():
    """Rebuild the FAISS index from all files in data/."""
    t0 = time.perf_counter()
    try:
        n = ingest_mod.main()
        _load_index()
        elapsed = int((time.perf_counter() - t0) * 1000)
        return IngestResponse(
            status="ok",
            documents_indexed=len(state.metadata),
            elapsed_ms=elapsed,
        )
    except Exception as e:
        log.exception("[INGEST] failed")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────
# POST /api/search
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/search", response_model=SearchResponse, tags=["search"])
def search_endpoint(req: SearchRequest):
    """Search documents and optionally classify the query."""
    if state.model is None or state.index is None:
        raise HTTPException(503, "Index not loaded. POST /api/ingest first.")

    t0 = time.perf_counter()
    k = req.top_k or settings.top_k

    docs = retrieve(req.query, state.model, state.index, state.metadata, top_k=k)
    classification = classify_from_results(docs) if req.classify else None
    answer = generate(req.query, docs) if req.generate_answer else None

    elapsed = int((time.perf_counter() - t0) * 1000)
    log.info("[SEARCH] q='%s' hits=%d ms=%d", req.query[:60], len(docs), elapsed)

    hits = [
        SearchHit(
            text=d.text[:500],
            score=d.score,
            source=d.source,
            category=d.category,
        )
        for d in docs
    ]

    clf_info = None
    if classification:
        clf_info = ClassificationInfo(
            predicted_category=classification.predicted_category,
            confidence=classification.confidence,
            category_scores=classification.category_scores,
        )

    return SearchResponse(
        query=req.query,
        hits=hits,
        classification=clf_info,
        answer=answer.answer if answer else None,
        answer_status=answer.status if answer else None,
        elapsed_ms=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /api/stats
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/stats", response_model=StatsResponse, tags=["meta"])
def stats_endpoint():
    """Return index statistics and category distribution."""
    if not state.metadata:
        raise HTTPException(503, "No index loaded.")

    cats = {}
    for d in state.metadata:
        cat = d.get("category", "") or "uncategorised"
        cats[cat] = cats.get(cat, 0) + 1

    sources = {}
    for d in state.metadata:
        src = d.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    return StatsResponse(
        total_documents=len(state.metadata),
        categories=cats,
        sources=sources,
        embed_model=ingest_mod.EMBED_MODEL_NAME,
        index_dimension=state.index.d if state.index else 0,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /api/evaluate
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/evaluate", response_model=EvalResponse, tags=["eval"])
def evaluate_endpoint():
    """Evaluate retrieval quality on labelled documents."""
    if state.model is None or state.index is None:
        raise HTTPException(503, "Index not loaded.")

    labelled = [d for d in state.metadata if d.get("category")]
    if len(labelled) < 3:
        raise HTTPException(400, "Need at least 3 labelled documents for eval.")

    queries = [d["text"][:100] for d in labelled]
    categories = [d["category"] for d in labelled]

    metrics = evaluate_retrieval(
        queries, categories, state.model, state.index, state.metadata
    )
    return EvalResponse(**metrics)


# ──────────────────────────────────────────────────────────────────────
# Static UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    index_html = UI_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse({"service": "semantic-search", "ui": "missing"})
