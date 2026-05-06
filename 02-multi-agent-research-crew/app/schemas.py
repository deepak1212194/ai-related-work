"""
schemas.py — API contracts
==========================
Multi-Agent Research API — App layer
"""

from typing import Literal

from pydantic import BaseModel, Field


class CrewRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)


class TraceEvent(BaseModel):
    """One event in the SSE stream — emitted at each agent boundary."""

    seq: int
    agent: Literal["planner", "researcher", "critic", "writer", "system"]
    role: str
    phase: Literal["start", "thinking", "tool", "output", "done"]
    content: str
    elapsed_ms: int


class CrewResponse(BaseModel):
    """Final non-streaming response (used by /api/crew/run)."""

    topic: str
    plan: str
    research: str
    critique: str
    final_brief: str
    total_elapsed_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_configured: bool
    llm_model: str
