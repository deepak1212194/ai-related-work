"""
main.py — FastAPI + SSE streaming
==================================
Multi-Agent Research API — App layer

Two endpoints expose the same crew:
  POST /api/crew/run     — synchronous, returns the full transcript
  GET  /api/crew/stream  — Server-Sent Events; emits each agent's output live

The SSE endpoint mirrors the live-streaming pattern used in real
multi-agent products (the GTC-demoed Building Permit Inspector being the
production reference). Each agent boundary becomes one SSE event so the
UI can render reasoning as it happens.

Run:
    uvicorn app.main:app --reload --port 8001
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import UI_DIR, settings
from .schemas import CrewRequest, CrewResponse, HealthResponse, TraceEvent
from src.agents import CREW
from src.orchestrator import CrewContext, _call_llm, _run_agent, run_crew
from src.tools import TOOLS

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] %s  (llm=%s)",
             settings.service_name, settings.llm_model)
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="Multi-Agent Research API",
    description="Sequentially-coordinated agents — planner, researcher, critic, writer.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Health
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    import os
    return HealthResponse(
        status="ok",
        llm_configured=bool(os.environ.get("OPENAI_API_KEY")),
        llm_model=settings.llm_model,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 2: Synchronous run (collects everything, returns once)
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/crew/run", response_model=CrewResponse, tags=["crew"])
def crew_run(req: CrewRequest) -> CrewResponse:
    t0 = time.perf_counter()
    ctx = run_crew(req.topic)
    return CrewResponse(
        topic=ctx.topic,
        plan=ctx.plan,
        research=ctx.research,
        critique=ctx.critique,
        final_brief=ctx.final_brief,
        total_elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


# ──────────────────────────────────────────────────────────────────────
# Module 3: SSE streaming — emits one event per agent boundary
# ──────────────────────────────────────────────────────────────────────
async def _stream_crew(topic: str) -> AsyncIterator[bytes]:
    """Async generator yielding SSE-framed TraceEvents."""
    seq = 0
    t0 = time.perf_counter()

    def fmt(ev: TraceEvent) -> bytes:
        return f"event: trace\ndata: {ev.model_dump_json()}\n\n".encode()

    async def emit(agent_name: str, role: str, phase: str, content: str) -> bytes:
        nonlocal seq
        seq += 1
        ev = TraceEvent(
            seq=seq,
            agent=agent_name,
            role=role,
            phase=phase,
            content=content,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
        return fmt(ev)

    yield await emit("system", "Crew Orchestrator", "start", f"Topic: {topic}")

    ctx = CrewContext(topic=topic)
    user_msgs = [
        f"Topic: {topic}",
        f"Topic: {topic}\n\nSubtasks:\n{{plan}}",
        f"Subtasks:\n{{plan}}\n\nResearch:\n{{research}}",
        f"Topic: {topic}\n\nResearch:\n{{research}}\n\nCritique:\n{{critique}}",
    ]
    fields = ["plan", "research", "critique", "final_brief"]

    for agent, tmpl, field_name in zip(CREW, user_msgs, fields):
        yield await emit(agent.name, agent.role, "thinking",
                         f"{agent.name} is working…")
        # Compose user message with already-known fields
        user = tmpl.format(
            plan=ctx.plan, research=ctx.research, critique=ctx.critique,
        )
        # Run the agent off the event loop so the server stays responsive
        out = await asyncio.to_thread(_run_agent, agent, user, ctx)
        setattr(ctx, field_name, out)
        yield await emit(agent.name, agent.role, "output", out)

    yield await emit("system", "Crew Orchestrator", "done",
                     "Crew complete.")


@app.get("/api/crew/stream", tags=["crew"])
async def crew_stream(topic: str) -> StreamingResponse:
    """SSE: live trace of each agent's output."""
    return StreamingResponse(
        _stream_crew(topic),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────────────
# Module 4: Static UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"ui": "missing"})
