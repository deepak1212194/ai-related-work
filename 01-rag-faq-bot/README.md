# 01 · RAG FAQ Service

A production-shaped **Retrieval-Augmented Generation** service with a **hallucination guard**, exposed over a typed HTTP API and a built-in chat UI.

> The LLM only generates when retrieval crosses a similarity floor. Below the floor, the service returns a structured `refused` response instead of fabricating.

## Highlights

- **Typed REST API** — FastAPI + Pydantic schemas for every request and response
- **Singleton model + index** — embedding model and FAISS index loaded once at startup, reused across requests (sub-100 ms p95 retrieval after warmup)
- **Built-in chat UI** at `/` — single-file HTML, dark/light auto, streams citations alongside answers
- **Hallucination guard** — `min_sim` floor + `refused` status so callers can branch on quality
- **Health endpoint** for liveness / readiness probes
- **Dockerised** — single container, healthcheck, non-root, persistent index volume
- **Configurable** — every knob (`top_k`, `min_sim`, model name) overrideable via `RAG_*` env vars

## Architecture

```
       ┌──────────┐                    ┌─────────────────────────────────┐
       │  Chat UI │  ─── /api/query ──▶│  FastAPI                        │
       │   (web)  │                    │  ├─ /api/query                  │
       └──────────┘                    │  ├─ /api/ingest                 │
                                       │  └─ /health                     │
                                       └────┬────────────────────────────┘
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                   ┌──────────┐      ┌──────────┐       ┌──────────┐
                   │ retrieve │      │ generate │       │  ingest  │
                   │ (FAISS)  │      │ (LLM +   │       │ (chunk + │
                   │          │      │  guard)  │       │  embed)  │
                   └──────────┘      └──────────┘       └──────────┘
```

## Project layout

```
01-rag-faq-bot/
├── app/                      # service layer (FastAPI)
│   ├── main.py               # routes + lifespan
│   ├── config.py             # pydantic-settings
│   ├── schemas.py            # request / response models
│   └── deps.py               # process-wide singletons
├── src/                      # core RAG logic (importable as a library)
│   ├── ingest.py             # chunk → embed → FAISS index
│   ├── retrieve.py           # top-K retrieval with score floor
│   ├── generate.py           # LLM call with constrained-context prompt
│   └── pipeline.py           # CLI end-to-end (still works standalone)
├── ui/
│   └── index.html            # chat interface
├── data/
│   └── sample_faqs.txt       # public, hand-written FAQs
├── tests/                    # pytest smoke tests
├── Dockerfile                # python:3.11-slim, healthcheck, single worker
├── docker-compose.yml        # one-command stack
└── requirements.txt
```

## Run locally

```bash
cd 01-rag-faq-bot
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Start the service (auto-builds index on first /api/ingest)
uvicorn app.main:app --reload --port 8000

# 2. Build the index
curl -X POST http://localhost:8000/api/ingest

# 3. Open the chat UI
open http://localhost:8000/                            # macOS
start http://localhost:8000/                           # Windows
```

## Run with Docker

```bash
cd 01-rag-faq-bot
OPENAI_API_KEY=sk-...   docker compose up --build      # optional key for real generation
```

The compose file mounts `./artifacts` so the FAISS index survives `docker compose down`.

## API reference

```
GET  /                  → chat UI
GET  /health            → readiness ({status, index_loaded, chunks_indexed, embed_model})
POST /api/ingest        → re-build the FAISS index from data/sample_faqs.txt
POST /api/query         → {question} → {status, answer, citations[], elapsed_ms}
```

OpenAPI spec auto-generated at `/docs` (Swagger) and `/redoc`.

## Configuration (env vars, all prefixed `RAG_`)

| Variable | Default | Notes |
|---|---|---|
| `RAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Swap for `all-mpnet-base-v2` for higher quality |
| `RAG_TOP_K` | `3` | Chunks retrieved per query |
| `RAG_MIN_SIM` | `0.45` | Score floor — below this, the service refuses |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `OPENAI_API_KEY` | *(unset)* | Without it, the service runs in offline-stub mode |

## Tests

```bash
pytest -q
```

The smoke test boots the FastAPI app in-process, re-indexes a tiny fixture, and asserts both the answered and refused paths.
