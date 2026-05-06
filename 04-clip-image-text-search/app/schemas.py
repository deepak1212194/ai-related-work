"""
schemas.py — CLIP service contracts
=====================================
"""

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=6, ge=1, le=50)


class Match(BaseModel):
    path: str
    score: float
    url: str          # browser-fetchable image URL


class SearchResponse(BaseModel):
    query: str
    matches: list[Match]
    elapsed_ms: int


class UploadResponse(BaseModel):
    status: Literal["ok"]
    filename: str
    size_bytes: int


class IndexBuildResponse(BaseModel):
    status: Literal["ok", "empty"]
    indexed: int
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    n_images: int
    index_built: bool
