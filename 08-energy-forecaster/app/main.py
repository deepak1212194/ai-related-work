"""
main.py — FastAPI application
==============================
Energy Forecasting Service

Endpoints:
  GET  /health        → service status
  POST /api/prepare   → run data preparation pipeline
  POST /api/train     → train models and compare
  GET  /api/results   → get latest comparison results
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

from .schemas import (
    PrepareResponse, TrainResponse, ResultsResponse, HealthResponse
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] Energy Forecasting Service ready")
    yield

app = FastAPI(
    title="Energy Forecasting Service",
    description="SARIMA vs XGBoost comparison for 24h-ahead electricity demand forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    has_data = (ARTIFACTS_DIR / "train.csv").exists()
    has_results = (ARTIFACTS_DIR / "model_comparison.json").exists()
    return HealthResponse(
        status="ok",
        data_prepared=has_data,
        models_trained=has_results,
    )


@app.post("/api/prepare", response_model=PrepareResponse, tags=["pipeline"])
def prepare_data():
    """Run the data preparation pipeline."""
    try:
        from src.data_prep import main as prep_main
        df, train, test, quality, bounds = prep_main()
        return PrepareResponse(
            status="ok",
            total_rows=len(df),
            train_rows=len(train),
            test_rows=len(test),
            anomaly_count=bounds.get("n_anomalies", 0),
            quality_report=quality,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.exception("Prepare failed")
        raise HTTPException(500, str(e))


@app.post("/api/train", response_model=TrainResponse, tags=["pipeline"])
def train_models():
    """Train SARIMA and XGBoost, return comparison."""
    import pandas as pd
    train_path = ARTIFACTS_DIR / "train.csv"
    test_path = ARTIFACTS_DIR / "test.csv"

    if not train_path.exists():
        raise HTTPException(400, "Data not prepared. POST /api/prepare first.")

    try:
        train = pd.read_csv(train_path, index_col="datetime", parse_dates=True)
        test = pd.read_csv(test_path, index_col="datetime", parse_dates=True)

        from src.model import compare_models
        comparison = compare_models(train, test)

        return TrainResponse(status="ok", comparison=comparison)
    except Exception as e:
        log.exception("Training failed")
        raise HTTPException(500, str(e))


@app.get("/api/results", response_model=ResultsResponse, tags=["results"])
def get_results():
    """Get latest model comparison results."""
    comp_path = ARTIFACTS_DIR / "model_comparison.json"
    bounds_path = ARTIFACTS_DIR / "anomaly_bounds.json"
    quality_path = ARTIFACTS_DIR / "data_quality_report.json"

    if not comp_path.exists():
        raise HTTPException(404, "No results yet. POST /api/train first.")

    comparison = json.loads(comp_path.read_text())
    bounds = json.loads(bounds_path.read_text()) if bounds_path.exists() else {}
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}

    return ResultsResponse(
        comparison=comparison,
        anomaly_bounds=bounds,
        data_quality=quality,
    )


@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"service": "energy-forecaster"})
