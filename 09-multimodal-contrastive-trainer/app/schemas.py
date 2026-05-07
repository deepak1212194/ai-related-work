"""Pydantic schemas for Multimodal Contrastive Trainer."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    history_available: bool


class TrainRequest(BaseModel):
    loss_type: str = Field("contrastive", pattern="^(contrastive|focal)$")
    num_epochs: int = Field(5, ge=1, le=100)
    batch_size: int = Field(16, ge=2, le=128)
    learning_rate: float = Field(1e-4, gt=0)


class TrainResponse(BaseModel):
    status: str
    epochs_completed: int
    history: List[Dict] = []
    eval_metrics: Dict = {}


class HistoryResponse(BaseModel):
    history: List[Dict] = []


class MetricsResponse(BaseModel):
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_rank: float = 0.0
    n_samples: int = 0
