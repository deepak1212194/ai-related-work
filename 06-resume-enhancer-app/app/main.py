"""
main.py — Resume Enhancer FastAPI app
=======================================

Two endpoints do the work:

  POST /api/enhance   — multipart upload of a .pdf or .tex resume.
                        Returns a job_id and final status, with two
                        download URLs (.tex and .pdf).
  GET  /api/result/{job_id}.tex — download the enhanced .tex
  GET  /api/result/{job_id}.pdf — download the enhanced .pdf
                                  (404 if compilation failed)

The UI at / is a single HTML page that drives both.
"""

import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .compiler import compile_pdf, render_tex
from .config import UI_DIR, WORK_DIR, settings
from .enhancer import count_sections, enhance
from .llm import get_llm
from .parser import parse_pdf, parse_tex
from .schemas import EnhanceResponse, HealthResponse

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log.info("[STARTUP] %s · backend=%s", settings.service_name, settings.llm_backend)
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="Resume Enhancer",
    description=(
        "Senior-AI-Engineer resume polisher. Uploads a PDF or .tex, "
        "applies the senior-ai-resume-craft enhancement rules, and "
        "returns both .tex and .pdf."
    ),
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
    backend_ok = True
    if settings.llm_backend == "anthropic" and not settings.anthropic_api_key:
        backend_ok = False
    pdflatex_ok = shutil.which(settings.pdflatex_cmd) is not None
    return HealthResponse(
        status="ok" if backend_ok else "degraded",
        backend=settings.llm_backend,
        backend_configured=backend_ok,
        pdflatex_available=pdflatex_ok,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 2: Enhance — the main endpoint
# ──────────────────────────────────────────────────────────────────────
ALLOWED_EXTS = {".pdf", ".tex"}


@app.post("/api/enhance", response_model=EnhanceResponse, tags=["enhance"])
async def enhance_resume(file: UploadFile = File(...)) -> EnhanceResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(415, f"Unsupported type: '{suffix}'. Send .pdf or .tex.")

    body = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(413, f"File > {settings.max_upload_size_mb} MB")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(body)

    t0 = time.perf_counter()

    # 1. Parse
    try:
        if suffix == ".pdf":
            parsed = parse_pdf(input_path)
        else:
            parsed = parse_tex(input_path)
    except Exception as e:                  # noqa: BLE001
        log.exception("[PARSE] failed")
        raise HTTPException(400, f"Failed to parse resume: {e}")

    # 2. Enhance via LLM
    try:
        llm = get_llm()
    except Exception as e:                  # noqa: BLE001
        raise HTTPException(503, str(e))

    parsed, notes = enhance(parsed, llm)

    # 3. Render .tex
    tex = render_tex(parsed)
    out_tex = job_dir / "resume.tex"
    out_tex.write_text(tex, encoding="utf-8")

    # 4. Compile to PDF (best-effort; .tex still usable on failure)
    pdf_path = compile_pdf(out_tex)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return EnhanceResponse(
        job_id=job_id,
        status="complete",
        sections_enhanced=count_sections(parsed),
        elapsed_ms=elapsed_ms,
        backend=llm.name,
        tex_path=f"/api/result/{job_id}.tex",
        pdf_path=f"/api/result/{job_id}.pdf" if pdf_path else None,
        pdf_compiled=pdf_path is not None,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 3: Downloads
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/result/{job_id}.tex", tags=["enhance"])
def download_tex(job_id: str) -> FileResponse:
    if not job_id.isalnum():
        raise HTTPException(400, "invalid job id")
    p = WORK_DIR / job_id / "resume.tex"
    if not p.exists():
        raise HTTPException(404, "tex not found for this job")
    return FileResponse(p, media_type="application/x-tex", filename="resume.tex")


@app.get("/api/result/{job_id}.pdf", tags=["enhance"])
def download_pdf(job_id: str) -> FileResponse:
    if not job_id.isalnum():
        raise HTTPException(400, "invalid job id")
    p = WORK_DIR / job_id / "resume.pdf"
    if not p.exists():
        raise HTTPException(
            404, "pdf not available — pdflatex may have failed or is not installed",
        )
    return FileResponse(p, media_type="application/pdf", filename="resume.pdf")


# ──────────────────────────────────────────────────────────────────────
# Module 4: UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"ui": "missing"})
