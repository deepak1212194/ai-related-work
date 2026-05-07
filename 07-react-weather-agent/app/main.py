"""
main.py — FastAPI application
==============================
ReAct Weather Agent

Endpoints:
  GET  /health       → service status
  POST /api/query    → run the agent (returns full trace)
  GET  /api/stream   → SSE stream of reasoning steps
  GET  /             → browser UI
"""

import json
import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path

from .schemas import AgentRequest, AgentResponse, StepInfo, HealthResponse
from src.agent import react_loop, run_agent, is_weather_query

# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] ReAct Weather Agent ready")
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="ReAct Weather Agent",
    description=(
        "A pure ReAct (Reasoning + Acting) agent that answers weather queries "
        "via iterative tool calling. Supports online (LLM) and offline modes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    return HealthResponse(
        status="ok",
        mode="online" if has_key else "offline",
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini") if has_key else "offline-mock",
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "6")),
    )


# ──────────────────────────────────────────────────────────────────────
# POST /api/query — full (non-streaming) agent execution
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/query", response_model=AgentResponse, tags=["agent"])
def query_endpoint(req: AgentRequest):
    """Run the ReAct agent and return the full trace + answer."""
    try:
        result = run_agent(req.question)
        return AgentResponse(
            query=result.query,
            answer=result.answer,
            status=result.status,
            total_iterations=result.total_iterations,
            steps=[
                StepInfo(
                    iteration=s.iteration,
                    step_type=s.step_type,
                    content=s.content,
                    tool_name=s.tool_name or None,
                    tool_args=s.tool_args or None,
                )
                for s in result.steps
            ],
        )
    except Exception as e:
        log.exception("Agent error")
        raise HTTPException(500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────
# GET /api/stream — SSE streaming of reasoning steps
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/stream", tags=["agent"])
async def stream_endpoint(question: str = Query(..., min_length=1)):
    """Stream ReAct steps as Server-Sent Events."""

    async def event_generator():
        gen = react_loop(question)
        try:
            while True:
                step = next(gen)
                payload = json.dumps({
                    "iteration": step.iteration,
                    "step_type": step.step_type,
                    "content": step.content,
                    "tool_name": step.tool_name,
                })
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.05)  # small delay for UI animation
        except StopIteration:
            yield "data: {\"step_type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────────────
# Static UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    index_html = UI_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse({"service": "react-weather-agent", "ui": "missing"})
