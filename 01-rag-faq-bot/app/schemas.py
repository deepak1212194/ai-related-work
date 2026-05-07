"""Pydantic schemas for the Semantic Search & Classification Service."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of results")
    classify: bool = Field(True, description="Run weighted-vote classification")
    generate_answer: bool = Field(False, description="Generate LLM answer (requires OPENAI_API_KEY)")


# ── Responses ────────────────────────────────────────────────────────

class SearchHit(BaseModel):
    text: str
    score: float
    source: str = ""
    category: str = ""


class ClassificationInfo(BaseModel):
    predicted_category: str
    confidence: float
    category_scores: Dict[str, float] = {}


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
    classification: Optional[ClassificationInfo] = None
    answer: Optional[str] = None
    answer_status: Optional[str] = None
    elapsed_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    embed_model: str
    documents_indexed: int


class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    elapsed_ms: int = 0
    message: str = ""


class StatsResponse(BaseModel):
    total_documents: int
    categories: Dict[str, int]
    sources: Dict[str, int]
    embed_model: str
    index_dimension: int


class EvalResponse(BaseModel):
    accuracy: float
    mrr: float
    recall_at_k: float
    k: int
    n_queries: int
