"""
app.py - FastAPI backend for the AI Resume Enhancer.

Replaces the Gradio interface with a fully-custom dark-theme single-page
HTML app served at GET / and driven by three JSON/SSE endpoints.

Run:
    python -m ui.app
    # → http://localhost:7860
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings, workdir      # noqa: E402
from app.core.llm import is_backend_configured     # noqa: E402
from app.core.skills import load_skills            # noqa: E402
from app.pipeline import ROLES, PipelineConfig, run_pipeline  # noqa: E402

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)

# Suppress Windows-only asyncio ProactorEventLoop noise that fires when the
# SSE stream connection closes while the pipeline is still running.
# Safe to ignore — the error is in socket cleanup, not application logic.
if sys.platform == "win32":
    class _SuppressPipeNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "_call_connection_lost" not in record.getMessage()
    logging.getLogger("asyncio").addFilter(_SuppressPipeNoise())

load_skills()

app = FastAPI(title="AI Resume Enhancer")

# in-memory job store: job_id → {q, result, error}
_jobs: dict[str, dict] = {}
_rate_store: dict[str, list[float]] = {}


def _check_rate(session_id: str) -> Optional[str]:
    now = time.time()
    store = _rate_store.setdefault(session_id, [])
    _rate_store[session_id] = [t for t in store if now - t < 3600]
    if len(_rate_store[session_id]) >= settings.max_runs_per_hour:
        wait = max(1, int((3600 - (now - _rate_store[session_id][0])) / 60))
        return f"Rate limit hit: {settings.max_runs_per_hour} runs/hr. Retry in ~{wait} min."
    _rate_store[session_id].append(now)
    return None


# ── HTML shell ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


# ── Start enhancement ─────────────────────────────────────────────────────────
@app.post("/api/enhance")
async def enhance(
    file: UploadFile         = File(...),
    role_id: str             = Form("ai_ml_engineer"),
    groq_api_key: str        = Form(""),
    hf_api_key: str          = Form(""),
    hf_model: str            = Form("mistralai/Mistral-7B-Instruct-v0.3"),
    hf_extraction_model: str = Form("mistralai/Mistral-7B-Instruct-v0.3"),
    groq_model: str          = Form("llama-3.1-8b-instant"),
    max_iter: int            = Form(3),
    session_id: str          = Form("default"),
):
    if not (file.filename or "").endswith(".tex"):
        return JSONResponse({"error": "Please upload a .tex file."}, status_code=400)

    rate_err = _check_rate(session_id)
    if rate_err:
        return JSONResponse({"error": rate_err}, status_code=429)

    if groq_api_key.strip():
        os.environ["GROQ_API_KEY"] = groq_api_key.strip()
    if groq_model.strip():
        os.environ["GROQ_MODEL"] = groq_model.strip()
    if hf_api_key.strip():
        os.environ["HF_API_KEY"] = hf_api_key.strip()
    if hf_model.strip():
        os.environ["HF_MODEL"] = hf_model.strip()
    if hf_extraction_model.strip():
        os.environ["HF_EXTRACTION_MODEL"] = hf_extraction_model.strip()

    if not (is_backend_configured("groq") or is_backend_configured("huggingface")):
        return JSONResponse(
            {"error": (
                "No API key provided. Add a free Groq key from console.groq.com/keys "
                "(recommended) or a HuggingFace token from huggingface.co/settings/tokens."
            )},
            status_code=400,
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        return JSONResponse(
            {"error": f"File too large. Max {settings.max_upload_kb} KB."},
            status_code=400,
        )

    job_id  = uuid.uuid4().hex[:12]
    job_dir = workdir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    tex_path = job_dir / (file.filename or "resume.tex")
    tex_path.write_bytes(contents)

    q: queue.Queue = queue.Queue()
    _jobs[job_id] = {"q": q, "result": None, "error": None}

    cfg = PipelineConfig(
        role_id=role_id,
        backend="auto",
        enable_critic=True,
        enable_role_review=True,
        enable_jd_matching=True,
        max_iterations=min(int(max_iter), 4),
    )

    def _worker() -> None:
        try:
            _jobs[job_id]["result"] = run_pipeline(
                tex_path, cfg,
                progress=lambda ev, data: q.put((ev, data), block=False),
                is_file=True,
            )
        except Exception as exc:
            log.exception("[app] pipeline crashed for job %s", job_id)
            _jobs[job_id]["error"] = str(exc)
        finally:
            q.put(("done", {}), block=False)

    threading.Thread(target=_worker, daemon=True, name=f"job-{job_id}").start()
    return {"job_id": job_id, "role_name": ROLES.get(role_id, role_id)}


# ── SSE progress stream ───────────────────────────────────────────────────────
@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "job not found"}, status_code=404)

    def _gen():
        q = _jobs[job_id]["q"]
        while True:
            try:
                event, data = q.get(timeout=180)
            except Exception:
                yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"
                return
            yield f"data: {json.dumps({'event': event, 'data': data})}\n\n"
            if event == "done":
                return

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Fetch final result ────────────────────────────────────────────────────────
@app.get("/api/result/{job_id}")
async def result(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "job not found"}, status_code=404)
    job = _jobs[job_id]
    if job["error"]:
        return JSONResponse({"error": job["error"]}, status_code=500)
    r = job["result"]
    if r is None:
        return JSONResponse({"status": "running"}, status_code=202)
    return _serialize(r)


# ── Download enhanced .tex ────────────────────────────────────────────────────
@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = _jobs.get(job_id)
    if not job or not job.get("result") or not job["result"].tex_path:
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(
        job["result"].tex_path,
        filename="enhanced_resume.tex",
        media_type="text/plain; charset=utf-8",
    )


# ── Result serialiser ─────────────────────────────────────────────────────────
def _serialize(r) -> dict:
    ats         = r.ats
    ats_score   = round(ats.score if ats else 0.0, 1)
    ats_matched = (ats.matched[:20]               if ats else [])
    ats_missing = (ats.missing_high_impact[:12]   if ats else [])

    jd           = r.jd_report
    jd_before    = round(jd.avg_score_before if jd else 0.0, 1)
    jd_after     = round(jd.avg_score_after  if jd else 0.0, 1)
    jd_delta     = round(jd.avg_delta        if jd else 0.0, 1)
    jd_gaps      = (jd.top_gaps[:8]          if jd else [])
    jd_samples   = []
    if jd:
        for s in jd.samples[:5]:
            jd_samples.append({
                "title":  s.title or f"JD {s.role_id}",
                "before": round(s.score_before, 1),
                "after":  round(s.score_after,  1),
                "delta":  round(s.delta,         1),
            })

    rv = r.role_reviews[0] if r.role_reviews else None

    traces = [
        {
            "label":   t.label,
            "before":  t.before,
            "after":   t.after,
            "changed": t.changed,
            "score":   round(t.final_score, 1),
            "iters":   t.iterations_used,
            "note":    t.note or "",
        }
        for t in r.section_traces
    ]

    tex_issues = [w for w in r.warnings if w.startswith("[LaTeX]")]
    return {
        "status":          r.status,
        "elapsed_s":       round(r.elapsed_ms / 1000, 1),
        "role":            ROLES.get(r.role, r.role),
        "warnings":        [w for w in r.warnings[:6] if not w.startswith("[LaTeX]")],
        "tex_issues":      [w.replace("[LaTeX] ", "") for w in tex_issues],
        "errors":          r.errors[:3],
        "ats_score":       ats_score,
        "ats_matched":     ats_matched,
        "ats_missing":     ats_missing,
        "jd_before":       jd_before,
        "jd_after":        jd_after,
        "jd_delta":        jd_delta,
        "jd_gaps":         jd_gaps,
        "jd_samples":      jd_samples,
        "role_score":      round(rv.overall_score if rv else 0.0, 1),
        "role_verdict":    rv.one_line_verdict   if rv else "",
        "role_strengths":  rv.strengths[:3]      if rv else [],
        "role_weaknesses": rv.weaknesses[:3]     if rv else [],
        "changed":         sum(1 for t in r.section_traces if t.changed),
        "total":           len(r.section_traces),
        "traces":          traces,
        "tex_content":     r.tex_content[:8000] if r.tex_content else "",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("RESUME_UI_PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
