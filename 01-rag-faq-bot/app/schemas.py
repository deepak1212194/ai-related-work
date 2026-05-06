"""
schemas.py — Public API contracts
==================================
RAG FAQ Service — App layer

All request and response shapes for the HTTP API live here so the
contract is one-stop-discoverable and validated at the edge.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# /api/query
# ──────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    score: float
    snippet: str


class QueryResponse(BaseModel):
    status: Literal["answered", "refused"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    elapsed_ms: int


# ──────────────────────────────────────────────────────────────────────
# /api/ingest
# ──────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    status: Literal["ok", "error"]
    chunks_indexed: int
    elapsed_ms: int
    message: str | None = None


# ──────────────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    embed_model: str
    chunks_indexed: int
