# 04 · CLIP Visual Search Service

A production-shaped **multimodal retrieval** service. Drop images via drag-drop, build a CLIP vector index, and search them with natural language — over a typed HTTP API, with a built-in gallery UI.

## Highlights

- **FastAPI service** with `/api/upload`, `/api/index/build`, `/api/search`, `/health`
- **Drag-drop UI** that uploads images, triggers index builds, and renders search results in a thumbnail grid with similarity scores
- **CLIP** (`openai/clip-vit-base-patch32`) cached as a process-wide singleton — image vectors are pre-computed once and persisted to a `.npz`
- **Static-mounted images** so result thumbnails load directly from the same service
- **Bounded uploads** (default 10 MB, MIME-validated)
- **Dockerised** with persistent volumes for images and artifacts

## Architecture

```
                ┌────────────────────────────────────────────┐
                │  FastAPI                                    │
   browser ────▶│  ├─ /api/upload     (multipart)             │
                │  ├─ /api/index/build                        │
                │  ├─ /api/search     (text query)            │
                │  ├─ /images/*       (static thumbnails)     │
                │  └─ /health                                 │
                └──────────────────┬──────────────────────────┘
                                   │
                            ┌──────┴──────┐
                            ▼             ▼
                    ┌─────────────┐  ┌──────────────┐
                    │ CLIP image  │  │ CLIP text    │
                    │ encoder     │  │ encoder      │
                    │ (batch)     │  │ (per query)  │
                    └─────────────┘  └──────────────┘
                            │
                            ▼
                    artifacts/image_vectors.npz
                    artifacts/manifest.json
```

## Project layout

```
04-clip-image-text-search/
├── app/
│   ├── main.py        # routes
│   ├── config.py      # settings
│   ├── schemas.py     # request/response models
│   └── deps.py        # CLIP model + vector cache
├── src/
│   ├── embed_images.py   # CLI for batch embedding
│   └── search.py         # CLI for one-shot search
├── ui/
│   └── index.html        # drag-drop + search gallery
├── data/
│   └── images/           # uploads land here (gitignored when populated)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run locally

```bash
cd 04-clip-image-text-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8002
open http://localhost:8002/                  # macOS / Linux
start http://localhost:8002/                 # Windows
```

In the UI:
1. Drag a few JPG/PNG images onto the drop zone (or click to choose).
2. Hit **⚙ Build index** — the service encodes them with CLIP.
3. Type a query — `"a photo of a sunset"`, `"a person on a beach"` — and hit **Search**. The grid shows the top matches with cosine-similarity scores.

## Run with Docker

```bash
docker compose up --build
```

Volumes mount `./data/images` and `./artifacts` so uploads + index survive restarts.

## API reference

```
GET  /                          → drag-drop gallery UI
GET  /health                    → readiness ({status, model, n_images, index_built})
POST /api/upload                → multipart file → {filename, size_bytes}
POST /api/index/build           → re-build the vector index → {indexed, elapsed_ms}
POST /api/search                → {query, top_k} → top-K matches with thumbnail URLs
GET  /images/{filename}         → static thumbnail
```

## Configuration (`CLIP_*` env vars)

| Variable | Default | Notes |
|---|---|---|
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | Swap for `clip-vit-large-patch14` for higher quality |
| `CLIP_DEFAULT_TOP_K` | `6` | Default `top_k` if request omits it |
| `CLIP_MAX_UPLOAD_SIZE_MB` | `10` | Per-image upload cap |
