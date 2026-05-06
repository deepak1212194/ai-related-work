"""
main.py — FastAPI application
==============================
RAG FAQ Service — App layer

Run locally:
    uvicorn app.main:app --reload --port 8000

Run via Docker:
    docker compose up --build
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import settings, UI_DIR
from .deps import state, warmup, load_index
from .schemas import (
    Citation,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src import generate as gen_mod
from src import ingest as ingest_mod
from src.retrieve import RetrievedChunk

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Lifespan: warm up on startup
# ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] %s", settings.service_name)
    warmup()
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="RAG FAQ Service",
    description="Retrieval-Augmented Generation with hallucination guard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Health
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.is_ready else "degraded",
        index_loaded=state.index is not None,
        embed_model=settings.embed_model,
        chunks_indexed=len(state.chunks),
    )


# ──────────────────────────────────────────────────────────────────────
# Module 2: Ingest
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/ingest", response_model=IngestResponse, tags=["index"])
def ingest() -> IngestResponse:
    """Re-build the FAISS index from `data/sample_faqs.txt`."""
    t0 = time.perf_counter()
    try:
        ingest_mod.main()
        load_index()                # refresh in-process state
        elapsed = int((time.perf_counter() - t0) * 1000)
        return IngestResponse(
            status="ok",
            chunks_indexed=len(state.chunks),
            elapsed_ms=elapsed,
        )
    except Exception as e:           # noqa: BLE001
        log.exception("[INGEST] failed")
        return JSONResponse(
            status_code=500,
            content=IngestResponse(
                status="error",
                chunks_indexed=0,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                message=str(e),
            ).model_dump(),
        )


# ──────────────────────────────────────────────────────────────────────
# Module 3: Query
# ──────────────────────────────────────────────────────────────────────
def _retrieve(question: str, top_k: int) -> List[RetrievedChunk]:
    """Per-request retrieval using the cached model + index."""
    if state.model is None or state.index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. POST /api/ingest first.",
        )

    qvec = state.model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, idxs = state.index.search(qvec, top_k)
    scores, idxs = scores[0], idxs[0]

    out: List[RetrievedChunk] = []
    for s, i in zip(scores, idxs):
        if i == -1 or s < settings.min_sim:
            continue
        out.append(RetrievedChunk(text=state.chunks[i], score=float(s)))
    return out


@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    k = req.top_k or settings.top_k

    chunks = _retrieve(req.question, k)
    answer = gen_mod.generate(req.question, chunks)

    elapsed = int((time.perf_counter() - t0) * 1000)
    log.info("[QUERY] q='%s' status=%s ms=%d",
             req.question[:60], answer.status, elapsed)

    return QueryResponse(
        status=answer.status,
        answer=answer.answer,
        citations=[
            Citation(score=c.score, snippet=c.text[:240])
            for c in chunks
        ],
        elapsed_ms=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 4: Static UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    index_html = UI_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse({"service": settings.service_name, "ui": "missing"})
