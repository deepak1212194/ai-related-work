# 02 · Multi-Agent Research API (live SSE)

A production-shaped multi-agent system with **four sequentially-coordinated specialists** (Planner → Researcher → Critic → Writer) — exposed over a typed HTTP API, with a **Server-Sent Events** endpoint that streams each agent's output to the browser in real time.

The architecture mirrors the live-streaming pattern used in real multi-agent products — most directly the Building Permit Inspector demonstrated at NVIDIA GTC 2026, where four CrewAI agents coordinate a compliance review on top of an LLM.

## Highlights

- **FastAPI service** with both blocking (`/api/crew/run`) and streaming (`/api/crew/stream`) endpoints
- **Server-Sent Events** — every agent boundary emits a `TraceEvent` with phase (`thinking`, `output`, `done`)
- **Built-in live UI** — left rail shows agent status with pulsing dots; right pane streams reasoning as it arrives
- **Tight role definition** — each agent has a focused system prompt (one file edit to tune)
- **Tool routing** — researcher is the only agent allowed to call tools (`web_search_stub`, `calculator`)
- **Offline mode** — without `OPENAI_API_KEY` the service runs deterministic stubs so the orchestration trace is always inspectable
- **Dockerised** with healthcheck

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │        FastAPI                              │
   browser ◀── SSE ─│  /api/crew/stream    /api/crew/run          │
                    └─────┬───────────────────────────┬───────────┘
                          │                           │
                          ▼                           ▼
                  ┌───────────────────┐      ┌──────────────────┐
                  │  Orchestrator     │      │  Synchronous run │
                  │  (yields events)  │      │  (returns once)  │
                  └────────┬──────────┘      └──────────────────┘
                           │
              ┌────────────┼─────────────┐──────────────┐
              ▼            ▼             ▼              ▼
       ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐
       │ Planner  │→│  Researcher  │→│  Critic  │→│  Writer  │
       └──────────┘ └──────┬───────┘ └──────────┘ └──────────┘
                           ▼
                       ┌────────┐
                       │ Tools  │
                       │ search │
                       │ calc   │
                       └────────┘
```

## Project layout

```
02-multi-agent-research-crew/
├── app/
│   ├── main.py             # FastAPI app + SSE generator
│   ├── config.py           # pydantic-settings
│   └── schemas.py          # CrewRequest, CrewResponse, TraceEvent
├── src/
│   ├── agents.py           # 4 frozen-dataclass agents (one prompt each)
│   ├── tools.py            # web_search_stub, safe calculator
│   ├── orchestrator.py     # sequential run + per-agent runner
│   └── main.py             # CLI entry (still works standalone)
├── ui/
│   └── index.html          # live SSE dashboard
├── examples/
│   └── sample_run.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run locally

```bash
cd 02-multi-agent-research-crew
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional — without it, the service runs in offline-stub mode
export OPENAI_API_KEY=sk-...

uvicorn app.main:app --reload --port 8001
open http://localhost:8001/                  # macOS / Linux
start http://localhost:8001/                 # Windows
```

Type a topic in the left panel and hit **Run crew (live)**. Each agent's output streams in below; agent dots on the left light up in sequence.

## Run with Docker

```bash
OPENAI_API_KEY=sk-...   docker compose up --build
```

## API reference

```
GET  /                          → live SSE dashboard
GET  /health                    → readiness
POST /api/crew/run              → {topic} → full transcript (blocking)
GET  /api/crew/stream?topic=…   → SSE; events of type "trace"
```

The SSE event payload (`TraceEvent`):

```json
{
  "seq": 4,
  "agent": "researcher",
  "role": "Researcher",
  "phase": "output",
  "content": "1. (offline) Mixture-of-Experts LLMs activate only…",
  "elapsed_ms": 6320
}
```

## Configuration (`CREW_*` env vars)

| Variable | Default | Notes |
|---|---|---|
| `CREW_LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CREW_LLM_TEMPERATURE` | `0.2` | Lower for determinism |
| `OPENAI_API_KEY` | *(unset)* | Without it → offline stub mode |
