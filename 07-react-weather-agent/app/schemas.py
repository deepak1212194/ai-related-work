"""Pydantic schemas for the ReAct Weather Agent."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question (weather-related)")


class StepInfo(BaseModel):
    iteration: int
    step_type: str  # "thought" | "action" | "observation" | "answer"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None


class AgentResponse(BaseModel):
    query: str
    answer: str
    status: str  # "success" | "rejected" | "max_iterations" | "error"
    total_iterations: int = 0
    steps: List[StepInfo] = []


class HealthResponse(BaseModel):
    status: str
    mode: str
    model: str
    max_iterations: int
