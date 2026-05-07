"""Pydantic schemas for Energy Forecasting Service."""
from typing import Dict, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    data_prepared: bool
    models_trained: bool


class PrepareResponse(BaseModel):
    status: str
    total_rows: int
    train_rows: int
    test_rows: int
    anomaly_count: int = 0
    quality_report: Dict = {}


class TrainResponse(BaseModel):
    status: str
    comparison: Dict = {}


class ResultsResponse(BaseModel):
    comparison: Dict = {}
    anomaly_bounds: Dict = {}
    data_quality: Dict = {}
