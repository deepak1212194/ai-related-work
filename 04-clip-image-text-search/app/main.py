"""
main.py — CLIP search service
==============================
CLIP Image-Text Search — App layer

Run:
    uvicorn app.main:app --reload --port 8002
"""

import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import UI_DIR, settings
from .deps import (
    MANIFEST_PATH,
    VEC_PATH,
    encode_text,
    load_index,
    state,
    warmup,
)
from .schemas import (
    HealthResponse,
    IndexBuildResponse,
    Match,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] %s", settings.service_name)
    warmup()
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="CLIP Visual Search",
    description="Multimodal retrieval: query an image set with text.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose the images folder under /images so the UI can render thumbnails
app.mount(
    "/images",
    StaticFiles(directory=settings.images_dir, check_dir=False),
    name="images",
)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Health
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    n = len(list(settings.images_dir.glob("*"))) if settings.images_dir.exists() else 0
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model=settings.model_name,
        n_images=n,
        index_built=state.index_built,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 2: Upload
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/upload", response_model=UploadResponse, tags=["index"])
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if Path(file.filename or "").suffix.lower() not in EXTS:
        raise HTTPException(415, f"Unsupported type: {file.filename}")

    body = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(413, f"File > {settings.max_upload_size_mb} MB")

    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    target = settings.images_dir / safe_name
    target.write_bytes(body)
    log.info("[UPLOAD] saved %s (%d bytes)", target, len(body))
    return UploadResponse(status="ok", filename=safe_name, size_bytes=len(body))


# ──────────────────────────────────────────────────────────────────────
# Module 3: Build index
# ──────────────────────────────────────────────────────────────────────
def _list_images() -> list[Path]:
    return sorted(
        p for p in settings.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    )


@app.post("/api/index/build", response_model=IndexBuildResponse, tags=["index"])
def build_index() -> IndexBuildResponse:
    t0 = time.perf_counter()
    paths = _list_images()
    if not paths:
        return IndexBuildResponse(status="empty", indexed=0,
                                  elapsed_ms=int((time.perf_counter() - t0) * 1000))

    assert state.model is not None and state.processor is not None
    BATCH = 8
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            images = [Image.open(p).convert("RGB") for p in chunk]
            inputs = state.processor(images=images, return_tensors="pt")
            feats = state.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_vecs.append(feats.cpu().numpy().astype("float32"))

    vecs = np.vstack(all_vecs)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VEC_PATH, vecs=vecs)
    MANIFEST_PATH.write_text(
        json.dumps({"paths": [p.name for p in paths]}, indent=2),
        encoding="utf-8",
    )
    load_index()    # refresh state
    elapsed = int((time.perf_counter() - t0) * 1000)
    log.info("[INDEX] indexed %d images in %d ms", len(paths), elapsed)
    return IndexBuildResponse(status="ok", indexed=len(paths), elapsed_ms=elapsed)


# ──────────────────────────────────────────────────────────────────────
# Module 4: Search
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    if not state.index_built:
        raise HTTPException(503, "Index not built. POST /api/index/build first.")

    t0 = time.perf_counter()
    qvec = encode_text(req.query)
    sims = (state.image_vecs @ qvec.T).squeeze()
    k = min(req.top_k, sims.shape[0])
    top_idx = np.argsort(-sims)[:k]

    matches = [
        Match(
            path=state.image_paths[i],
            score=float(sims[i]),
            url=f"/images/{state.image_paths[i]}",
        )
        for i in top_idx
    ]
    elapsed = int((time.perf_counter() - t0) * 1000)
    log.info("[SEARCH] q='%s' top1=%.3f ms=%d",
             req.query[:60], matches[0].score if matches else 0.0, elapsed)
    return SearchResponse(query=req.query, matches=matches, elapsed_ms=elapsed)


# ──────────────────────────────────────────────────────────────────────
# Module 5: UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"ui": "missing"})
