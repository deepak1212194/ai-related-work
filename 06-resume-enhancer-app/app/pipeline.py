"""
pipeline.py - top-level resume enhancement pipeline.

Pipeline stages:
  1. Parse        - .tex -> ResumeIR  (deterministic)
  2. Repair       - ExtractorAgent fills missed fields (LLM, optional)
  3. Complete     - CompleterAgent inserts placeholders for required gaps
  4. Plan         - PlannerAgent picks section/skill/experience order
  5. Enhance      - per-section EnhancerAgent + CriticAgent loop
  6. Render       - IR -> .tex string
  7. Score        - ATSReport + JDMatchReport (current role)
  8. Cross-validate - JDMatchReport for each of the 5 cross-validation roles
  9. Role review  - RoleReviewerAgent for each of the 5 roles (LLM)

Production hardening:
  - File size validation and LaTeX sanitization on upload
  - Dynamic protected terms extracted from the user's resume
  - Skills context passed to enhancer for keyword grounding
  - Work directory TTL cleanup
  - Explicit path vs content handling (no heuristic guessing)
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set

from .agents.completer import CompleterAgent
from .agents.critic import CriticAgent
from .agents.enhancer import EnhancerAgent
from .agents.extractor import ExtractorAgent
from .agents.jd_matcher import JDMatchAgent, load_role_jds
from .agents.orchestrator import IterativeOrchestrator
from .agents.planner import PlannerAgent, apply_plan
from .agents.role_reviewer import RoleReviewerAgent
from .core.ats import score_keywords
from .core.config import DANGEROUS_TEX_COMMANDS, settings, workdir
from .core.ir import (
    ATSReport, JDMatchReport, PipelineResult, ResumeIR, RoleReview,
    SectionTrace,
)
from .core.llm import LLMClient, LLMError, build_llm, is_backend_configured
from .core.safety import extract_protected_terms_from_ir, get_all_protected_terms
from .core.skills import get_bundle, load_skills
from .parser import parse_tex_to_ir
from .render import render_ir_to_tex
from .render.renderer import validate_tex


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Role registry - exposed to the UI
# ----------------------------------------------------------------------
ROLES = {
    "ai_ml_engineer": "AI / ML Engineer",
    "data_scientist": "Data Scientist",
    "software_engineer": "Software Engineer",
    "devops_cloud_engineer": "DevOps / Cloud Engineer",
    "product_manager": "Product Manager",
}


def list_role_keywords(role_id: str) -> List[str]:
    """Pulled from the role's skill .md `priority_keywords` block."""
    bundle = get_bundle()
    sf = bundle.role_files.get(role_id)
    if not sf:
        return []
    block = sf.blocks.get("priority_keywords", "") or sf.default
    if not block:
        return []
    out: List[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s or s.startswith("##"):
            continue
        for tok in s.split(","):
            tok = tok.strip().rstrip(".").rstrip(",")
            if tok and len(tok) <= 60:
                out.append(tok)
    # de-dupe preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for tok in out:
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tok)
    return deduped


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def validate_tex_content(content: str) -> List[str]:
    """Scan .tex content for dangerous LaTeX commands.

    Returns a list of warning messages. If any are critical, the caller
    should reject the upload.
    """
    warnings: List[str] = []
    content_lower = content.lower()
    for cmd in DANGEROUS_TEX_COMMANDS:
        if cmd.lower() in content_lower:
            warnings.append(
                f"Potentially dangerous LaTeX command detected: {cmd}. "
                "This command has been flagged for security review."
            )
    return warnings


def validate_file_size(path: Path) -> Optional[str]:
    """Check file size against the configured limit.

    Returns an error message if too large, else None.
    """
    size = path.stat().st_size
    if size > settings.max_upload_bytes:
        return (
            f"File is too large ({size // 1024}KB). "
            f"Maximum allowed size is {settings.max_upload_kb}KB."
        )
    return None


# ----------------------------------------------------------------------
# Work directory cleanup
# ----------------------------------------------------------------------
def cleanup_old_jobs() -> int:
    """Delete job directories older than the configured TTL.

    Returns the number of directories cleaned up.
    """
    work = workdir()
    ttl_s = settings.work_dir_ttl_hours * 3600
    cutoff = time.time() - ttl_s
    cleaned = 0
    try:
        for entry in work.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
                if mtime < cutoff:
                    import shutil
                    shutil.rmtree(entry, ignore_errors=True)
                    cleaned += 1
            except Exception:                               # noqa: BLE001
                pass
    except Exception:                                       # noqa: BLE001
        pass
    if cleaned:
        log.info("[pipeline] cleaned %d old job directories", cleaned)
    return cleaned


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
ProgressCallback = Callable[[str, dict], None]


@dataclass
class PipelineConfig:
    role_id: str = "ai_ml_engineer"
    backend: str = "auto"
    enable_critic: bool = True
    enable_role_review: bool = True       # review the target role only
    enable_jd_matching: bool = True
    enable_cross_role: bool = False        # opt-in: also score against the other 4 roles
    max_iterations: int = 3
    enable_multi_llm: bool = True
    max_section_calls: int = 120


def _emit(cb: Optional[ProgressCallback], event: str, data: dict) -> None:
    if cb is None:
        return
    try:
        cb(event, data)
    except Exception:                                       # noqa: BLE001
        log.debug("[pipeline] progress callback raised; ignoring")


def _bullet_id(blk_idx: int, kind: str, sub_idx: int = 0, group_idx: int = -1) -> str:
    if group_idx >= 0:
        return f"exp_{blk_idx}_g{group_idx}_{kind}_{sub_idx}"
    return f"exp_{blk_idx}_{kind}_{sub_idx}"


def _build_skills_context(ir: ResumeIR) -> str:
    """Build a compact skills context string for keyword grounding."""
    parts: List[str] = []
    for bucket in ir.skills:
        parts.append(f"{bucket.name}: {', '.join(bucket.items)}")
    return "\n".join(parts)


_ACTION_VERBS = {
    "architected", "engineered", "designed", "built", "owned", "led",
    "optimized", "deployed", "delivered", "implemented", "scaled",
    "automated", "integrated", "refactored", "productionized",
}
_IMPACT_TOKENS = {
    "%", "latency", "accuracy", "throughput", "users", "cost", "faster",
    "reduced", "improved", "decrease", "increase", "sla",
}


def _should_skip_rewrite(text: str, mode: str) -> bool:
    """Heuristic skip to reduce low-value LLM calls in fast modes."""
    if mode == "accuracy":
        return False
    t = (text or "").strip()
    if len(t) < 60 or len(t) > 360:
        return False
    low = t.lower()
    has_verb = any(v in low for v in _ACTION_VERBS)
    has_impact = any(tok in low for tok in _IMPACT_TOKENS) or bool(re.search(r"\d", low))
    has_tech = bool(re.search(r"\b(ai|ml|llm|rag|python|pytorch|docker|kubernetes|azure|aws|gcp|api)\b", low))
    return has_verb and has_impact and has_tech


def _is_resume_like(ir: ResumeIR) -> bool:
    """Guardrail: reject non-resume .tex files early."""
    required_hits = 0
    if ir.header.name:
        required_hits += 1
    if ir.summary:
        required_hits += 1
    if ir.skills:
        required_hits += 1
    if ir.experience:
        required_hits += 1
    if ir.education:
        required_hits += 1
    return required_hits >= 2


def _count_rewrite_candidates(ir: ResumeIR, mode: str) -> int:
    """Count per-bullet rewrite candidates (used for rough token estimates)."""
    count = 0
    if ir.summary and not _looks_like_placeholder(ir.summary) and not _should_skip_rewrite(ir.summary, mode):
        count += 1
    for blk in ir.experience:
        if blk.placeholder:
            continue
        for bullet in blk.bullets:
            if bullet.strip() and not _should_skip_rewrite(bullet, mode):
                count += 1
        for grp in blk.groups:
            for bullet in grp.bullets:
                if bullet.strip() and not _should_skip_rewrite(bullet, mode):
                    count += 1
    for proj in ir.projects:
        if proj.placeholder:
            continue
        for bullet in proj.bullets:
            if bullet.strip() and not _should_skip_rewrite(bullet, mode):
                count += 1
    if mode != "speed":
        for bucket in ir.skills:
            if not bucket.placeholder and bucket.items:
                count += 1
    for ach in ir.achievements:
        if not ach.placeholder and ach.description.strip() and not _should_skip_rewrite(ach.description, mode):
            count += 1
    return count


def _count_rewrite_blocks(ir: ResumeIR, mode: str) -> int:
    """Count block-level orchestrator units — one per experience/project block.

    This matches how the pipeline now emits progress events (one per block,
    not one per bullet), so the UI percentage stays accurate.
    """
    count = 0
    if ir.summary and not _looks_like_placeholder(ir.summary) and not _should_skip_rewrite(ir.summary, mode):
        count += 1
    for blk in ir.experience:
        if blk.placeholder:
            continue
        has_direct = any(b.strip() and not _should_skip_rewrite(b, mode) for b in blk.bullets)
        if has_direct:
            count += 1
        for grp in blk.groups:
            if any(b.strip() and not _should_skip_rewrite(b, mode) for b in grp.bullets):
                count += 1
    for proj in ir.projects:
        if proj.placeholder:
            continue
        if any(b.strip() and not _should_skip_rewrite(b, mode) for b in proj.bullets):
            count += 1
    if mode != "speed":
        for bucket in ir.skills:
            if not bucket.placeholder and bucket.items:
                count += 1
    for ach in ir.achievements:
        if not ach.placeholder and ach.description.strip() and not _should_skip_rewrite(ach.description, mode):
            count += 1
    return count


def run_pipeline(
    tex_input: str | Path,
    cfg: PipelineConfig,
    *,
    progress: Optional[ProgressCallback] = None,
    is_file: bool = True,
) -> PipelineResult:
    """Run the full enhancement pipeline on a .tex resume.

    Args:
        tex_input: Either a Path to a .tex file (when is_file=True) or
                   a string of .tex content (when is_file=False).
        cfg: Pipeline configuration.
        progress: Optional callback for live UI updates.
        is_file: True if tex_input is a file path, False if content string.
    """
    started = time.perf_counter()
    job_id = uuid.uuid4().hex[:12]
    job_dir = workdir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old job directories on each run
    cleanup_old_jobs()

    # Reload skills so any live edits to skills/*.md are picked up
    load_skills()

    result = PipelineResult(
        job_id=job_id,
        status="error",
        role=cfg.role_id,
        backend=cfg.backend,
        elapsed_ms=0,
    )

    # Build the LLM client (may raise LLMError)
    try:
        llm = build_llm(cfg.backend)
    except LLMError as e:
        result.errors.append(str(e))
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    # Optional second model for parse/repair: cheaper Hugging Face model.
    extraction_llm: LLMClient = llm
    extraction_model = os.environ.get("HF_EXTRACTION_MODEL", settings.hf_extraction_model)
    if cfg.enable_multi_llm and settings.enable_multi_llm and is_backend_configured("huggingface"):
        try:
            extraction_llm = build_llm("huggingface", model_override=extraction_model)
            _emit(progress, "stage", {
                "name": "llm_split",
                "status": "done",
                "extractor_backend": f"huggingface:{extraction_model}",
            })
        except Exception as e:                                  # noqa: BLE001
            result.warnings.append(f"multi-llm extraction disabled: {e}")

    bundle = get_bundle()
    extractor = ExtractorAgent(llm=extraction_llm, skills=bundle)
    completer = CompleterAgent(llm=llm, skills=bundle)
    planner = PlannerAgent(llm=llm, skills=bundle)
    enhancer = EnhancerAgent(llm=llm, skills=bundle)
    critic = CriticAgent(llm=llm, skills=bundle) if cfg.enable_critic else None
    reviewer = RoleReviewerAgent(llm=llm, skills=bundle)
    jd_agent = JDMatchAgent()

    # ----- 1. Parse (LLM-first; regex fallback) -----
    _emit(progress, "stage", {"name": "parse", "status": "start"})
    raw_excerpt = ""
    tex_path_obj: Optional[Path] = None
    llm_extracted = False

    if is_file:
        tex_path_obj = Path(tex_input)
        size_err = validate_file_size(tex_path_obj)
        if size_err:
            result.errors.append(size_err)
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            return result
        try:
            raw_excerpt = tex_path_obj.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            result.errors.append(f"Could not read file: {e}")
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            return result
        tex_warnings = validate_tex_content(raw_excerpt)
        if tex_warnings:
            result.warnings.extend(tex_warnings)
    else:
        raw_excerpt = str(tex_input)
        tex_warnings = validate_tex_content(raw_excerpt)
        if tex_warnings:
            result.warnings.extend(tex_warnings)

    # Try LLM-first extraction — handles any .tex template format
    ir: Optional[ResumeIR] = None
    try:
        ir = extractor.extract_full(raw_excerpt)
        if ir is not None and ir.header.name and (ir.experience or ir.skills or ir.education):
            llm_extracted = True
        else:
            ir = None
    except Exception as e:                                  # noqa: BLE001
        log.warning("[pipeline] LLM-first extraction failed: %s", e)
        ir = None

    # Fallback to regex parser
    if ir is None:
        if tex_path_obj is not None:
            ir = parse_tex_to_ir(tex_path_obj)
        else:
            from .parser.tex_parser import parse_tex_string
            ir = parse_tex_string(raw_excerpt)

    # Remove empty or duplicate extra sections so the renderer stays clean
    _KNOWN_EXTRA_TITLES = {
        "summary", "professional summary", "career summary", "objective",
        "profile", "about me", "about",
        "skills", "technical skills", "core competencies", "key skills", "expertise",
        "experience", "professional experience", "work experience", "employment",
        "employment history", "career history",
        "projects", "selected projects", "personal projects", "academic projects",
        "open source", "side projects",
        "education", "academic background", "qualifications",
        "certifications", "licenses & certifications", "courses & certifications",
        "courses and certifications", "credentials",
        "certifications and awards", "licenses and certifications",
        "achievements", "achievements & recognition", "honors & awards",
        "honors", "awards",
        "publications", "papers", "patents",
    }
    ir.extras = [
        x for x in ir.extras
        if (x.items or x.body.strip())
        and x.title.strip().lower() not in _KNOWN_EXTRA_TITLES
    ]

    result.ir_before = ir.model_copy(deep=True)
    if not _is_resume_like(ir):
        result.errors.append(
            "This .tex file does not look like a resume or is missing too many required fields. "
            "Please upload a resume .tex file with at least basic identity and section content."
        )
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result
    _emit(progress, "stage", {
        "name": "parse", "status": "done",
        "method": "llm" if llm_extracted else "regex",
        "name_extracted": ir.header.name,
        "experience_blocks": len(ir.experience),
        "skill_buckets": len(ir.skills),
        "education_blocks": len(ir.education),
        "projects": len(ir.projects),
        "achievements": len(ir.achievements),
        "certifications": len(ir.certifications),
        "publications": len(ir.publications),
    })

    # ----- 2. Repair (LLM) — skipped when LLM already did full extraction -----
    _emit(progress, "stage", {"name": "repair", "status": "start"})
    if not llm_extracted:
        try:
            ir = extractor.repair(ir, raw_excerpt)
        except Exception as e:                              # noqa: BLE001
            log.warning("[pipeline] repair stage failed: %s", e)
            result.warnings.append(f"repair stage skipped: {e}")
    _emit(progress, "stage", {"name": "repair", "status": "done"})

    # ----- 3. Complete -----
    _emit(progress, "stage", {"name": "complete", "status": "start"})
    ir = completer.fill(ir)
    _emit(progress, "stage", {
        "name": "complete", "status": "done",
        "completed_fields": ir.completed_fields,
    })

    # ----- 4. Plan -----
    _emit(progress, "stage", {"name": "plan", "status": "start"})
    plan = planner.plan(ir, cfg.role_id)
    apply_plan(ir, plan)
    _emit(progress, "stage", {
        "name": "plan", "status": "done",
        "section_order": ir.section_order,
    })

    # ----- Build dynamic protected terms and skills context -----
    protected_terms = get_all_protected_terms(ir)
    skills_context = _build_skills_context(ir)

    # Create orchestrator with dynamic protected terms
    orchestrator = IterativeOrchestrator(
        enhancer=enhancer, critic=critic,
        max_iterations=cfg.max_iterations,
        protected_terms=protected_terms,
    )

    # ----- 5. Enhance per section -----
    role_keywords = list_role_keywords(cfg.role_id)
    section_traces: List[SectionTrace] = []
    # Use block-level count so progress events match the new one-per-block emits
    total_rewrite_units = _count_rewrite_blocks(ir, "accuracy")
    _emit(progress, "stage", {
        "name": "enhance_plan",
        "status": "done",
        "total_units": total_rewrite_units,
        "mode": "accuracy",
    })

    # Summary
    if ir.summary and not _looks_like_placeholder(ir.summary) and not _should_skip_rewrite(ir.summary, "accuracy"):
        _emit(progress, "section", {"label": "Summary", "status": "start"})
        trace = orchestrator.run(
            section_id="summary",
            label="Professional Summary",
            section_type="summary",
            original=ir.summary,
            role_id=cfg.role_id,
            priority_keywords=role_keywords,
            skills_context=skills_context,
        )
        if trace.changed:
            ir.summary = trace.after
        section_traces.append(trace)
        _emit(progress, "section", {
            "label": "Summary", "status": "done",
            "score": trace.final_score, "changed": trace.changed,
            "before": trace.before[:500], "after": trace.after[:500],
            "note": trace.note,
        })

    section_calls = 0

    # Experience bullets — one block-level LLM call per experience entry
    for bi, blk in enumerate(ir.experience):
        if blk.placeholder:
            continue
        lead_bullets = set(plan.lead_bullet_hints.get(bi, []))

        # ---- direct bullets on the block ----
        eligible = [
            (ji, bullet)
            for ji, bullet in enumerate(blk.bullets)
            if bullet.strip() and not _should_skip_rewrite(bullet, "accuracy")
        ]
        if eligible:
            if section_calls >= cfg.max_section_calls:
                result.warnings.append(
                    "section call budget reached; remaining experience bullets kept as-is"
                )
            else:
                block_context = (
                    f"{blk.company or 'Unknown'} — {blk.title or 'Unknown'}"
                    + (f" ({blk.dates})" if blk.dates else "")
                )
                orig_indices   = [ji for ji, _ in eligible]
                orig_bullets   = [b  for _, b  in eligible]
                blk_labels     = [f"Bullet {bi+1}.{ji+1} — {blk.title or blk.company}" for ji, _ in eligible]
                blk_ids        = [_bullet_id(bi, "bullet", ji) for ji, _ in eligible]
                # Remap lead_bullets (original indices) to local indices
                lead_local     = {li for li, (ji, _) in enumerate(eligible) if ji in lead_bullets}
                blk_label      = f"Block {bi+1} — {blk.company or blk.title} ({len(eligible)} bullets)"
                _emit(progress, "section", {"label": blk_label, "status": "start"})
                traces = orchestrator.run_block(
                    block_id=f"exp_{bi}",
                    block_context=block_context,
                    section_type="bullet",
                    originals=orig_bullets,
                    labels=blk_labels,
                    section_ids=blk_ids,
                    role_id=cfg.role_id,
                    priority_keywords=role_keywords,
                    lead_indices=lead_local,
                    skills_context=skills_context,
                )
                for (ji, _), trace in zip(eligible, traces):
                    if trace.changed:
                        blk.bullets[ji] = trace.after
                    section_traces.append(trace)
                section_calls += 1
                avg_score = sum(t.final_score for t in traces) / len(traces) if traces else 0.0
                changed_traces = [t for t in traces if t.changed]
                sample_before = changed_traces[0].before[:300] if changed_traces else ""
                sample_after = changed_traces[0].after[:300] if changed_traces else ""
                _emit(progress, "section", {
                    "label": blk_label, "status": "done",
                    "score": avg_score, "changed": any(t.changed for t in traces),
                    "changed_count": len(changed_traces), "total_count": len(traces),
                    "sample_before": sample_before, "sample_after": sample_after,
                })

        # ---- grouped bullets (sub-roles / workstreams) ----
        for gi, group in enumerate(blk.groups):
            g_eligible = [
                (ji, bullet)
                for ji, bullet in enumerate(group.bullets)
                if bullet.strip() and not _should_skip_rewrite(bullet, "accuracy")
            ]
            if not g_eligible:
                continue
            if section_calls >= cfg.max_section_calls:
                result.warnings.append(
                    "section call budget reached; remaining grouped bullets kept as-is"
                )
                break
            grp_context   = f"{blk.company or 'Unknown'} — {group.label}"
            grp_orig_idx  = [ji for ji, _ in g_eligible]
            grp_bullets   = [b  for _, b  in g_eligible]
            grp_labels    = [f"Bullet {bi+1}.{gi+1}.{ji+1} — {group.label}" for ji, _ in g_eligible]
            grp_ids       = [_bullet_id(bi, "bullet", ji, gi) for ji, _ in g_eligible]
            grp_label     = f"Block {bi+1}.{gi+1} — {group.label} ({len(g_eligible)} bullets)"
            _emit(progress, "section", {"label": grp_label, "status": "start"})
            grp_traces = orchestrator.run_block(
                block_id=f"exp_{bi}_g{gi}",
                block_context=grp_context,
                section_type="bullet",
                originals=grp_bullets,
                labels=grp_labels,
                section_ids=grp_ids,
                role_id=cfg.role_id,
                priority_keywords=role_keywords,
                skills_context=skills_context,
            )
            for (ji, _), trace in zip(g_eligible, grp_traces):
                if trace.changed:
                    group.bullets[ji] = trace.after
                section_traces.append(trace)
            section_calls += 1
            avg_score = sum(t.final_score for t in grp_traces) / len(grp_traces) if grp_traces else 0.0
            grp_changed = [t for t in grp_traces if t.changed]
            _emit(progress, "section", {
                "label": grp_label, "status": "done",
                "score": avg_score, "changed": any(t.changed for t in grp_traces),
                "sample_before": grp_changed[0].before[:300] if grp_changed else "",
                "sample_after": grp_changed[0].after[:300] if grp_changed else "",
            })

    # Project bullets — one block-level call per project
    for pi, proj in enumerate(ir.projects):
        if proj.placeholder:
            continue
        p_eligible = [
            (ji, bullet)
            for ji, bullet in enumerate(proj.bullets)
            if bullet.strip() and not _should_skip_rewrite(bullet, "accuracy")
        ]
        if not p_eligible:
            continue
        if section_calls >= cfg.max_section_calls:
            result.warnings.append("section call budget reached; remaining project bullets kept as-is")
            break
        proj_context  = f"Project: {proj.name}"
        proj_orig_idx = [ji for ji, _ in p_eligible]
        proj_bullets  = [b  for _, b  in p_eligible]
        proj_labels   = [f"Project {pi+1}.{ji+1} — {proj.name}" for ji, _ in p_eligible]
        proj_ids      = [f"proj_{pi}_bullet_{ji}" for ji, _ in p_eligible]
        proj_label    = f"Project {pi+1} — {proj.name} ({len(p_eligible)} bullets)"
        _emit(progress, "section", {"label": proj_label, "status": "start"})
        proj_traces = orchestrator.run_block(
            block_id=f"proj_{pi}",
            block_context=proj_context,
            section_type="project_bullet",
            originals=proj_bullets,
            labels=proj_labels,
            section_ids=proj_ids,
            role_id=cfg.role_id,
            priority_keywords=role_keywords,
            skills_context=skills_context,
        )
        for (ji, _), trace in zip(p_eligible, proj_traces):
            if trace.changed:
                proj.bullets[ji] = trace.after
            section_traces.append(trace)
        section_calls += 1
        avg_score = sum(t.final_score for t in proj_traces) / len(proj_traces) if proj_traces else 0.0
        proj_changed = [t for t in proj_traces if t.changed]
        _emit(progress, "section", {
            "label": proj_label, "status": "done",
            "score": avg_score, "changed": any(t.changed for t in proj_traces),
            "sample_before": proj_changed[0].before[:300] if proj_changed else "",
            "sample_after": proj_changed[0].after[:300] if proj_changed else "",
        })

    # Skills - reorder/tighten each bucket
    for si, bucket in enumerate(ir.skills):
        if bucket.placeholder or not bucket.items:
            continue
        original_text = ", ".join(bucket.items)
        label = f"Skills - {bucket.name}"
        if section_calls >= cfg.max_section_calls:
            result.warnings.append("section call budget reached; remaining skill buckets kept as-is")
            break
        _emit(progress, "section", {"label": label, "status": "start"})
        trace = orchestrator.run(
            section_id=f"skills_{si}",
            label=label,
            section_type="skills",
            original=original_text,
            role_id=cfg.role_id,
            priority_keywords=role_keywords,
        )
        if trace.changed:
            # Parse the rewritten comma-separated items back
            new_items = [
                item.strip() for item in trace.after.split(",")
                if item.strip()
            ]
            if new_items:
                bucket.items = new_items
        section_traces.append(trace)
        section_calls += 1
        _emit(progress, "section", {
            "label": label, "status": "done",
            "score": trace.final_score, "changed": trace.changed,
            "sample_before": trace.before[:300], "sample_after": trace.after[:300],
        })

    # Achievements - rewrite the description only
    for ai, ach in enumerate(ir.achievements):
        if ach.placeholder or not ach.description.strip():
            continue
        if _should_skip_rewrite(ach.description, "accuracy"):
            continue
        label = f"Achievement {ai+1} - {ach.title}"
        if section_calls >= cfg.max_section_calls:
            result.warnings.append("section call budget reached; remaining achievements kept as-is")
            break
        _emit(progress, "section", {"label": label, "status": "start"})
        trace = orchestrator.run(
            section_id=f"ach_{ai}",
            label=label,
            section_type="achievement",
            original=ach.description,
            role_id=cfg.role_id,
            priority_keywords=role_keywords,
        )
        if trace.changed:
            ach.description = trace.after
        section_traces.append(trace)
        section_calls += 1
        _emit(progress, "section", {
            "label": label, "status": "done",
            "score": trace.final_score, "changed": trace.changed,
            "sample_before": trace.before[:300], "sample_after": trace.after[:300],
        })

    result.section_traces = section_traces
    result.ir_after = ir.model_copy(deep=True)

    # ----- 6. Render -----
    _emit(progress, "stage", {"name": "render", "status": "start"})
    try:
        tex = render_ir_to_tex(ir)
    except Exception as e:                                  # noqa: BLE001
        log.exception("[pipeline] render failed")
        result.errors.append(f"render failed: {e}")
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result
    out_tex = job_dir / "resume.tex"
    out_tex.write_text(tex, encoding="utf-8")
    result.tex_path = str(out_tex)
    result.tex_content = tex
    tex_issues = validate_tex(tex)
    for issue in tex_issues:
        result.warnings.append(f"[LaTeX] {issue}")
    _emit(progress, "stage", {
        "name": "render", "status": "done",
        "tex_path": str(out_tex),
        "char_count": len(tex),
        "tex_issues": tex_issues,
        "tex_preview": tex[:3000],   # first 3K chars for live preview pane
    })

    # ----- 7. ATS scoring (current role keywords) -----
    try:
        text_after = ir.text_blob()
        text_before = (result.ir_before.text_blob() if result.ir_before else "")
        ats = score_keywords(text_after, role_keywords)
        result.ats = ATSReport(
            score=ats.score,
            matched_count=ats.matched_count,
            total_checked=ats.total_checked,
            matched=ats.matched[:30],
            missing_high_impact=ats.missing[:20],
            suggestions=ats.suggestions,
        )
    except Exception as e:                                   # noqa: BLE001
        log.warning("[pipeline] ATS scoring failed: %s", e)
        result.warnings.append(f"ATS scoring skipped: {e}")
        text_after = ""
        text_before = ""

    # ----- 8. JD matching (target role; optional cross-role) -----
    if cfg.enable_jd_matching:
        try:
            jd_report = jd_agent.evaluate(
                role_id=cfg.role_id, text_before=text_before, text_after=text_after,
            )
            result.jd_report = jd_report
            _emit(progress, "stage", {
                "name": "jd_match", "status": "done",
                "avg_delta": jd_report.avg_delta, "samples": jd_report.samples_count,
            })
            if cfg.enable_cross_role:
                for rid in settings.cross_validate_roles:
                    if rid == cfg.role_id:
                        continue
                    r = jd_agent.evaluate(
                        role_id=rid, text_before=text_before, text_after=text_after,
                    )
                    if r.samples_count > 0:
                        result.cross_role_jd_reports.append(r)
        except Exception as e:                              # noqa: BLE001
            log.warning("[pipeline] JD matching failed: %s", e)
            result.warnings.append(f"JD matching skipped: {e}")

    # ----- 9. Role review (LLM, target role only by default) -----
    if cfg.enable_role_review:
        try:
            rname = ROLES.get(cfg.role_id, cfg.role_id)
            _emit(progress, "review", {"role": cfg.role_id, "status": "start"})
            review = reviewer.review(
                resume_text=text_after, role_id=cfg.role_id, role_name=rname,
            )
            result.role_reviews.append(review)
            _emit(progress, "review", {
                "role": cfg.role_id, "status": "done", "score": review.overall_score,
            })
            if cfg.enable_cross_role:
                for rid in settings.cross_validate_roles:
                    if rid == cfg.role_id:
                        continue
                    rname = ROLES.get(rid, rid)
                    _emit(progress, "review", {"role": rid, "status": "start"})
                    r = reviewer.review(
                        resume_text=text_after, role_id=rid, role_name=rname,
                    )
                    result.role_reviews.append(r)
                    _emit(progress, "review", {
                        "role": rid, "status": "done", "score": r.overall_score,
                    })
        except Exception as e:                              # noqa: BLE001
            log.warning("[pipeline] role review failed: %s", e)
            result.warnings.append(f"role review skipped: {e}")

    # ----- finalize -----
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    result.status = "complete" if not result.errors else (
        "partial" if result.tex_path else "error"
    )
    return result


def _looks_like_placeholder(s: str) -> bool:
    """Check if a string looks like a [PLACEHOLDER] token.

    More conservative than before: requires uppercase content inside
    brackets to avoid false positives with legitimate bracket usage.
    """
    stripped = s.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    inner = stripped[1:-1].strip()
    # Must be short and mostly uppercase to be a placeholder
    return len(inner) < 100 and inner == inner.upper()
