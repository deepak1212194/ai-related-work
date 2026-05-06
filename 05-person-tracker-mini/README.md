# 05 · Edge Person Tracker Service

A real-time multi-person tracking service: **YOLOv8 detection** + a **hand-rolled IoU-based tracker** + a **WebSocket streaming endpoint** that pushes per-frame bounding boxes to a browser, drawn live on top of an HTML5 `<video>` element.

The architecture mirrors the pattern I use for production edge-CV pipelines (RTSP camera streams + detection + tracker + alerting), packaged here as a self-contained service.

## Highlights

- **FastAPI service** with a video-upload endpoint, a static `/videos/` mount, and a `/api/track/stream` **WebSocket** that emits `FrameUpdate` JSON messages per frame
- **Live-overlay UI** — a `<canvas>` is sized to match the `<video>` element and re-drawn on each WebSocket message; bounding boxes follow people while the video plays
- **Modular core** — `detector` (YOLO wrapper) and `tracker` (IoU greedy matcher) are each importable, replaceable, and unit-testable
- **Tunable** — IoU threshold, max-missed frames, detection confidence, and frame-skip stride all live in `app/config.py` and accept env overrides
- **Singleton model** — YOLO is loaded once on startup, reused for every WebSocket session
- **Dockerised** with FFmpeg for broad codec support

## Architecture

```
                  ┌──────────────────────────────────────────────────┐
                  │  FastAPI                                          │
   browser ──HTTP─│  POST /api/upload     (stores video on disk)      │
   browser ──WS───│  /api/track/stream    (sends video_id, gets JSON) │
                  │  GET  /videos/<file>  (static playback)           │
                  │  GET  /health                                     │
                  └─────┬────────────────────────────────┬────────────┘
                        │                                │
                        ▼                                ▼
                ┌───────────────┐                ┌──────────────┐
                │   Detector    │                │   Tracker    │
                │   (YOLOv8n)   │  ── matches ──▶│  IoU greedy  │
                │               │                │  + ageing    │
                └───────────────┘                └──────┬───────┘
                                                        │
                                                  per-frame JSON
                                                        │
                                                        ▼
                                                ┌─────────────┐
                                                │  Browser    │
                                                │  canvas     │
                                                │  overlay    │
                                                └─────────────┘
```

## Project layout

```
05-person-tracker-mini/
├── app/
│   ├── main.py              # routes + WebSocket
│   ├── config.py            # all knobs
│   └── schemas.py           # FrameUpdate, TrackBox, …
├── src/
│   ├── detector.py          # YOLO wrapper
│   ├── tracker.py           # IoU greedy tracker with ageing
│   └── main.py              # CLI batch tool (still works standalone)
├── ui/
│   └── index.html           # video + canvas overlay + WS client
├── data/
│   └── uploads/             # uploaded videos (gitignored when populated)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run locally

```bash
cd 05-person-tracker-mini
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8004
open http://localhost:8004/                  # macOS / Linux
start http://localhost:8004/                 # Windows
```

In the UI:
1. Drag a video onto the drop zone.
2. Click **▶ Start tracking**. The video plays back on the left while the WebSocket streams per-frame bounding boxes; the canvas above redraws on every frame, so IDs follow people in real time.
3. Stats below the video update live: current frame, active tracks, total unique IDs seen.

## Run with Docker

```bash
docker compose up --build
```

The compose file mounts `./data/uploads` so your videos survive restarts.

## API reference

```
GET   /                          → live tracking UI
GET   /health                    → readiness ({status, model, iou_threshold, max_missed})
POST  /api/upload                → multipart video → {video_id, filename, size_bytes}
WS    /api/track/stream          → stream of FrameUpdate JSON per frame
GET   /videos/{filename}         → static video for playback
```

`FrameUpdate` payload:

```json
{
  "frame": 47,
  "timestamp_s": 1.88,
  "n_tracks": 3,
  "tracks": [
    { "track_id": 1, "x1": 412.3, "y1": 89.7, "x2": 537.6, "y2": 442.1 },
    { "track_id": 2, "x1": 22.0,  "y1": 110.0, "x2": 130.5, "y2": 460.2 }
  ]
}
```

## Configuration (`TRACKER_*` env vars)

| Variable | Default | Notes |
|---|---|---|
| `TRACKER_YOLO_MODEL` | `yolov8n.pt` | Tiny by default; swap for `yolov8s.pt` for accuracy |
| `TRACKER_DETECT_CONF` | `0.4` | Detection confidence floor |
| `TRACKER_IOU_THRESHOLD` | `0.30` | Track ↔ detection match floor |
| `TRACKER_MAX_MISSED_FRAMES` | `15` | Frames a track can survive without a match |
| `TRACKER_STREAM_EVERY_N_FRAMES` | `1` | Frame-skip — set 2/3 for faster playback at cost of detail |
