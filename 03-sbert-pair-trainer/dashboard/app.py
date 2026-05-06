"""
app.py — Metrics dashboard
==========================
SBERT Pair Trainer — Dashboard (FastAPI + Chart.js)

Serves a small dashboard at / that reads artifacts/metrics.json and
renders the train/eval metrics as cards plus a small bar chart.

Usage:
    uvicorn dashboard.app:app --port 8003
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
INDEX_HTML = BASE_DIR / "index.html"

app = FastAPI(title="SBERT Metrics Dashboard")


# ──────────────────────────────────────────────────────────────────────
# Module 1: API
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics() -> JSONResponse:
    if not METRICS_PATH.exists():
        return JSONResponse(
            {"status": "missing",
             "message": "Run `python -m pipeline.run` first."},
            status_code=404,
        )
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    payload["status"] = "ok"
    return JSONResponse(payload)


@app.get("/api/registry")
def registry() -> JSONResponse:
    """List all registered models on disk."""
    reg_root = PROJECT_ROOT / "artifacts" / "registry"
    if not reg_root.exists():
        return JSONResponse({"models": []})

    models = []
    for d in sorted(reg_root.iterdir()):
        if not d.is_dir():
            continue
        m = d / "metrics.json"
        eval_m = {}
        if m.exists():
            try:
                eval_m = json.loads(m.read_text(encoding="utf-8")).get("eval", {})
            except json.JSONDecodeError:
                pass
        models.append({"name": d.name, "eval": eval_m})
    return JSONResponse({"models": models})


# ──────────────────────────────────────────────────────────────────────
# Module 2: Static UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse(INDEX_HTML) if INDEX_HTML.exists() else JSONResponse(
        {"ui": "missing"}, status_code=404,
    )
