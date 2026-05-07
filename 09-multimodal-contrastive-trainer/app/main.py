"""
main.py — FastAPI application
==============================
Multimodal Contrastive Trainer

Endpoints:
  GET  /health        → status
  POST /api/train     → train the model
  GET  /api/history   → training history
  GET  /api/metrics   → evaluation metrics
  GET  /              → dashboard UI
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .schemas import HealthResponse, TrainRequest, TrainResponse, MetricsResponse, HistoryResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] Multimodal Contrastive Trainer ready")
    yield

app = FastAPI(
    title="Multimodal Contrastive Trainer",
    description="Dual-encoder (ResNet-50 + BERT) contrastive learning with InfoNCE and focal loss.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    has_model = (ARTIFACTS_DIR / "model.pt").exists()
    has_history = (ARTIFACTS_DIR / "train_history.json").exists()
    return HealthResponse(status="ok", model_trained=has_model, history_available=has_history)


@app.post("/api/train", response_model=TrainResponse, tags=["training"])
def train_endpoint(req: TrainRequest):
    """Train the multimodal model."""
    try:
        from src.model import train_model
        result = train_model(
            loss_type=req.loss_type,
            num_epochs=req.num_epochs,
            batch_size=req.batch_size,
            lr=req.learning_rate,
        )
        if result.status == "error":
            raise HTTPException(500, result.error)
        return TrainResponse(
            status=result.status,
            epochs_completed=result.epochs_completed,
            history=result.history,
            eval_metrics=result.eval_metrics or {},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Training failed")
        raise HTTPException(500, str(e))


@app.get("/api/history", response_model=HistoryResponse, tags=["results"])
def history_endpoint():
    """Get training history."""
    path = ARTIFACTS_DIR / "train_history.json"
    if not path.exists():
        raise HTTPException(404, "No training history. POST /api/train first.")
    history = json.loads(path.read_text())
    return HistoryResponse(history=history)


@app.get("/api/metrics", response_model=MetricsResponse, tags=["results"])
def metrics_endpoint():
    """Get evaluation metrics."""
    path = ARTIFACTS_DIR / "eval_metrics.json"
    if not path.exists():
        raise HTTPException(404, "No metrics. POST /api/train first.")
    metrics = json.loads(path.read_text())
    return MetricsResponse(**metrics)


@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"service": "multimodal-trainer"})
