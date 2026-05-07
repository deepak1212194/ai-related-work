"""
main.py — Resume Enhancer FastAPI app
=======================================

Endpoints:
  GET  /health                     liveness, lists available roles
  GET  /api/roles                  role catalog (used by the UI)
  GET  /api/skills                 skill file loading status
  POST /api/skills/reload          hot-reload skill files
  POST /api/enhance                multipart upload + role → enhanced
  POST /api/ats-score              compute ATS keyword score for text
  GET  /api/result/{job}.tex       download .tex
  GET  /api/result/{job}.pdf       download .pdf (404 if compile failed)
  GET  /                           drag-drop UI

Skill-file-driven: all enhancement rules come from editable
markdown files in the skills/ directory. Edit .md files to change
agent behavior — no code changes needed.

The enhance endpoint is GUARANTEED to return a structured response.
On any internal error it returns an EnhanceResponse with status="error"
and a human-readable warning, never a stack trace.
"""

import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .compiler import compile_pdf, render_tex
from .config import UI_DIR, WORK_DIR, settings
from .enhancer import count_sections, enhance, compute_ats_score
from .llm import LLMError, get_llm
from .parser import parse_pdf, parse_tex
from .rules import ROLE_PROFILES, list_roles
from .schemas import (
    ATSScore, EnhanceResponse, HealthResponse, RoleInfo,
    SkillInfoResponse,
)
from .skill_loader import get_skill_info, load_skills, reload_skills

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # Load skill files at startup
    skills = load_skills()
    log.info(
        "[STARTUP] %s · backend=%s · roles=%d · skill_files=%d",
        settings.service_name,
        settings.llm_backend,
        len(skills.roles),
        len(skills.loaded_files),
    )
    yield
    log.info("[SHUTDOWN]")


app = FastAPI(
    title="Resume Enhancer",
    description=(
        "Skill-file-driven resume enhancement agent. Rules are loaded "
        "from editable markdown files (skills/*.md) — edit them to change "
        "agent behavior without code changes. Supports 5 role profiles, "
        "ATS keyword scoring, and before/after previews."
    ),
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Health + role catalog + skill info
# ──────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    backend_ok = True
    if settings.llm_backend == "anthropic" and not settings.anthropic_api_key:
        backend_ok = False
    pdflatex_ok = shutil.which(settings.pdflatex_cmd) is not None
    skill_info = get_skill_info()
    return HealthResponse(
        status="ok" if backend_ok else "degraded",
        backend=settings.llm_backend,
        backend_configured=backend_ok,
        pdflatex_available=pdflatex_ok,
        available_roles=[RoleInfo(**r) for r in list_roles()],
        skill_files_loaded=len(skill_info.get("loaded_files", [])),
    )


@app.get("/api/roles", response_model=list[RoleInfo], tags=["meta"])
def get_roles() -> list[RoleInfo]:
    return [RoleInfo(**r) for r in list_roles()]


@app.get("/api/skills", response_model=SkillInfoResponse, tags=["skills"])
def skills_info() -> SkillInfoResponse:
    """Show which skill files are loaded and their status."""
    info = get_skill_info()
    return SkillInfoResponse(**info)


@app.post("/api/skills/reload", response_model=SkillInfoResponse, tags=["skills"])
def skills_reload() -> SkillInfoResponse:
    """Hot-reload all skill files from disk. Useful after editing .md files."""
    # Clear the ROLE_PROFILES cache too
    ROLE_PROFILES.clear()
    reload_skills()
    info = get_skill_info()
    log.info("[SKILL] Reloaded: %s", info)
    return SkillInfoResponse(**info)


# ──────────────────────────────────────────────────────────────────────
# Module 2: Enhance — never crashes, always returns a response
# ──────────────────────────────────────────────────────────────────────
ALLOWED_EXTS = {".pdf", ".tex"}


def _error_response(job_id: str, role: str, started: float,
                    backend: str, message: str) -> EnhanceResponse:
    return EnhanceResponse(
        job_id=job_id,
        status="error",
        role=role,
        sections_enhanced=0,
        sections_total=0,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        backend=backend,
        warnings=[message],
    )


@app.post("/api/enhance", response_model=EnhanceResponse, tags=["enhance"])
async def enhance_resume(
    file: UploadFile = File(...),
    role: str = Form(default=None),
) -> EnhanceResponse:
    job_id = uuid.uuid4().hex[:12]
    role_id = role or settings.default_role
    if role_id not in ROLE_PROFILES:
        # Try loading profiles if empty
        if not ROLE_PROFILES:
            from .rules import _ensure_loaded
            _ensure_loaded()
        if role_id not in ROLE_PROFILES:
            role_id = settings.default_role
    started = time.perf_counter()
    backend_label = settings.llm_backend

    # --- Validate input file ---
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTS:
        return _error_response(
            job_id, role_id, started, backend_label,
            f"Unsupported file type '{suffix}'. Please upload a .pdf or .tex.",
        )

    body = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(body) > max_bytes:
        return _error_response(
            job_id, role_id, started, backend_label,
            f"File is {len(body) // 1024} KB, max allowed is "
            f"{settings.max_upload_size_mb} MB.",
        )
    if len(body) < 100:
        return _error_response(
            job_id, role_id, started, backend_label,
            "File is too small to be a real resume (< 100 bytes).",
        )

    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(body)

    # --- Parse ---
    try:
        parsed = parse_pdf(input_path) if suffix == ".pdf" else parse_tex(input_path)
    except Exception as e:                              # noqa: BLE001
        log.warning("[PARSE] failed: %s", e)
        return _error_response(
            job_id, role_id, started, backend_label,
            f"Could not parse this file as a resume: {e!s}. "
            "If the PDF is scanned, please run OCR first or upload a .tex.",
        )

    sections_total = count_sections(parsed)
    if sections_total == 0:
        return _error_response(
            job_id, role_id, started, backend_label,
            "No recognizable resume sections found "
            "(Summary / Skills / Experience / Education / Achievements). "
            "Please check that the document has these standard sections.",
        )

    # --- Get LLM ---
    try:
        llm = get_llm()
    except LLMError as e:
        return _error_response(job_id, role_id, started, backend_label, str(e))

    backend_label = llm.name

    # --- Enhance ---
    try:
        parsed, notes, previews, warnings = enhance(parsed, llm, role_id)
    except Exception as e:                              # noqa: BLE001
        log.exception("[ENHANCE] unexpected error")
        return _error_response(
            job_id, role_id, started, backend_label,
            f"Enhancement loop raised an unexpected error and was "
            f"stopped: {e!s}. Your file is unchanged.",
        )

    # --- ATS Score ---
    ats_data = None
    try:
        # Build full text from parsed resume for ATS scoring
        full_text_parts = [parsed.summary or ""]
        for bucket, items in parsed.skills.items():
            full_text_parts.append(f"{bucket}: {items}")
        for block in parsed.experience_blocks:
            full_text_parts.extend(block.bullets)
        full_text = " ".join(full_text_parts)

        role_keywords = None
        profile = ROLE_PROFILES.get(role_id)
        if profile:
            role_keywords = profile.keywords

        ats_result = compute_ats_score(full_text, role_keywords)
        ats_data = ATSScore(**ats_result)
    except Exception as e:                              # noqa: BLE001
        log.warning("[ATS] scoring failed: %s", e)

    # --- Render .tex ---
    try:
        tex = render_tex(parsed)
    except Exception as e:                              # noqa: BLE001
        log.exception("[RENDER] template failed")
        return _error_response(
            job_id, role_id, started, backend_label,
            f"Template rendering failed: {e!s}.",
        )
    out_tex = job_dir / "resume.tex"
    out_tex.write_text(tex, encoding="utf-8")

    # --- Compile PDF (best-effort) ---
    pdf_path = None
    try:
        pdf_path = compile_pdf(out_tex)
    except Exception as e:                              # noqa: BLE001
        log.warning("[COMPILE] %s", e)
        warnings.append(f"PDF compilation skipped: {e!s}")

    if pdf_path is None and shutil.which(settings.pdflatex_cmd) is None:
        warnings.append(
            "pdflatex is not installed in this environment — only .tex was "
            "produced. Use Overleaf or install TeX Live to compile."
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    status = "complete" if not warnings else "partial"

    return EnhanceResponse(
        job_id=job_id,
        status=status,
        role=role_id,
        sections_enhanced=count_sections(parsed),
        sections_total=sections_total,
        elapsed_ms=elapsed_ms,
        backend=backend_label,
        tex_path=f"/api/result/{job_id}.tex",
        pdf_path=f"/api/result/{job_id}.pdf" if pdf_path else None,
        pdf_compiled=pdf_path is not None,
        notes=notes,
        previews=previews,
        warnings=warnings,
        ats_score=ats_data,
    )


# ──────────────────────────────────────────────────────────────────────
# Module 3: Standalone ATS scoring
# ──────────────────────────────────────────────────────────────────────
class ATSRequest(BaseModel):
    text: str
    role: str | None = None


@app.post("/api/ats-score", response_model=ATSScore, tags=["ats"])
def ats_score_endpoint(req: ATSRequest) -> ATSScore:
    """Compute ATS keyword score for any text. Useful for quick checks."""
    role_keywords = None
    if req.role and req.role in ROLE_PROFILES:
        role_keywords = ROLE_PROFILES[req.role].keywords
    result = compute_ats_score(req.text, role_keywords)
    return ATSScore(**result)


# ──────────────────────────────────────────────────────────────────────
# Module 4: Downloads
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
# Module 5: UI
# ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_ui():
    f = UI_DIR / "index.html"
    return FileResponse(f) if f.exists() else JSONResponse({"ui": "missing"})
