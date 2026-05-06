"""
main.py — Edge Person Tracker service
======================================
Person Tracker Mini — App layer

Two ways to consume tracking output:
  POST /api/track         — runs end-to-end, writes annotated MP4, returns path
  WS   /api/track/stream  — streams per-frame JSON updates as the detector runs

The WebSocket path is what the UI uses to draw bounding-box overlays on
top of an HTML5 <video> element while the original video plays back.

Run:
    uvicorn app.main:app --reload --port 8004
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import UI_DIR, settings
from .schemas import FrameUpdate, HealthResponse, TrackBox, UploadResponse
from src.detector import PersonDetector
from src.tracker import IouTracker

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ──────────────────────────────────────────────────────────────────────
# Detector singleton (loading YOLO is slow)
# ──────────────────────────────────────────────────────────────────────
_detector: PersonDetector | None = None


def get_detector() -> PersonDetector:
    global _detector
    if _detector is None:
        log.info("[STARTUP] Loading YOLO: %s", settings.yolo_model)
        _detector = PersonDetector(
            model_path=settings.yolo_model, conf=settings.detect_conf,
        )
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    log.info("[STARTUP] %s", settings.service_name)
    get_detector()  # warm
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="Edge Person Tracker",
    description="YOLO + IoU tracker as a service.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded videos for the UI's <video> element
app.mount(
    "/videos",
    StaticFiles(directory=settings.uploads_dir, check_dir=False),
    name="videos",
)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Health
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _detector is not None else "degraded",
        model=settings.yolo_model,
        iou_threshold=settings.iou_threshold,
        max_missed=settings.max_missed_frames,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 2: Upload
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/upload", response_model=UploadResponse, tags=["video"])
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in EXTS:
        raise HTTPException(415, f"Unsupported video type: {suffix}")

    body = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(413, f"Video > {settings.max_upload_size_mb} MB")

    video_id = uuid.uuid4().hex[:12]
    target = settings.uploads_dir / f"{video_id}{suffix}"
    target.write_bytes(body)
    log.info("[UPLOAD] saved %s (%d bytes)", target, len(body))
    return UploadResponse(
        status="ok", video_id=video_id, filename=target.name, size_bytes=len(body),
    )


# ──────────────────────────────────────────────────────────────────────
# Module 3: WebSocket streaming
# ──────────────────────────────────────────────────────────────────────
@app.websocket("/api/track/stream")
async def track_stream(ws: WebSocket):
    """
    Client sends `{video_id}` after connect; server streams FrameUpdate
    JSON for each frame it processes, then closes with a `done` envelope.
    """
    await ws.accept()
    try:
        msg = await ws.receive_json()
        video_id = msg.get("video_id")
        if not video_id:
            await ws.close(code=4400, reason="missing video_id")
            return

        match = next(settings.uploads_dir.glob(f"{video_id}.*"), None)
        if not match:
            await ws.close(code=4404, reason="video not found")
            return

        log.info("[WS] streaming tracker for %s", match.name)
        detector = get_detector()
        tracker = IouTracker(
            iou_threshold=settings.iou_threshold,
            max_missed=settings.max_missed_frames,
        )

        cap = cv2.VideoCapture(str(match))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        await ws.send_json({"event": "start", "fps": fps})

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % settings.stream_every_n_frames != 0:
                continue

            # Run detection in a thread so the event loop stays responsive
            dets = await asyncio.to_thread(detector.detect, frame)
            tracks = tracker.update(dets)

            update = FrameUpdate(
                frame=frame_idx,
                timestamp_s=frame_idx / fps,
                n_tracks=len(tracks),
                tracks=[
                    TrackBox(
                        track_id=t.track_id,
                        x1=t.bbox[0], y1=t.bbox[1],
                        x2=t.bbox[2], y2=t.bbox[3],
                    )
                    for t in tracks
                ],
            )
            await ws.send_text(update.model_dump_json())

        cap.release()
        await ws.send_json({"event": "done", "frames": frame_idx})
        log.info("[WS] done — %d frames", frame_idx)
    except WebSocketDisconnect:
        log.info("[WS] client disconnected")
    except Exception as e:                        # noqa: BLE001
        log.exception("[WS] error")
        try:
            await ws.send_json({"event": "error", "detail": str(e)})
        except Exception:                         # noqa: BLE001
            pass
    finally:
        try:
            await ws.close()
        except Exception:                         # noqa: BLE001
            pass


# ──────────────────────────────────────────────────────────────────────
# Module 4: UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"ui": "missing"})
