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
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .agents.completer import CompleterAgent
from .agents.critic import CriticAgent
from .agents.enhancer import EnhancerAgent
from .agents.extractor import ExtractorAgent
from .agents.jd_matcher import JDMatchAgent, load_role_jds
from .agents.orchestrator import IterativeOrchestrator
from .agents.planner import PlannerAgent, apply_plan
from .agents.role_reviewer import RoleReviewerAgent
from .core.ats import score_keywords
from .core.config import SKILLS_DIR, settings, workdir
from .core.ir import (
    ATSReport, JDMatchReport, PipelineResult, ResumeIR, RoleReview,
    SectionTrace,
)
from .core.llm import LLMClient, LLMError, build_llm
from .core.skills import get_bundle, load_skills
from .parser import parse_tex_to_ir
from .render import render_ir_to_tex


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


def run_pipeline(
    tex_input: str | Path,
    cfg: PipelineConfig,
    *,
    progress: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """Run the full enhancement pipeline on a .tex resume.

    `tex_input` may be a Path to a .tex file or a string of .tex content.
    """
    started = time.perf_counter()
    job_id = uuid.uuid4().hex[:12]
    job_dir = workdir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

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

    bundle = get_bundle()
    extractor = ExtractorAgent(llm=llm, skills=bundle)
    completer = CompleterAgent(llm=llm, skills=bundle)
    planner = PlannerAgent(llm=llm, skills=bundle)
    enhancer = EnhancerAgent(llm=llm, skills=bundle)
    critic = CriticAgent(llm=llm, skills=bundle) if cfg.enable_critic else None
    reviewer = RoleReviewerAgent(llm=llm, skills=bundle)
    jd_agent = JDMatchAgent()

    orchestrator = IterativeOrchestrator(
        enhancer=enhancer, critic=critic,
        max_iterations=cfg.max_iterations,
    )

    # ----- 1. Parse -----
    _emit(progress, "stage", {"name": "parse", "status": "start"})
    if isinstance(tex_input, Path) or (
        isinstance(tex_input, str) and len(tex_input) < 4096
        and Path(tex_input).suffix.lower() == ".tex"
        and Path(tex_input).exists()
    ):
        ir = parse_tex_to_ir(Path(tex_input))
        try:
            raw_excerpt = Path(tex_input).read_text(encoding="utf-8", errors="replace")
        except Exception:
            raw_excerpt = ""
    else:
        from .parser.tex_parser import parse_tex_string
        ir = parse_tex_string(str(tex_input))
        raw_excerpt = str(tex_input)
    result.ir_before = ir.model_copy(deep=True)
    _emit(progress, "stage", {
        "name": "parse", "status": "done",
        "name_extracted": ir.header.name,
        "experience_blocks": len(ir.experience),
        "skill_buckets": len(ir.skills),
    })

    # ----- 2. Repair (LLM) -----
    _emit(progress, "stage", {"name": "repair", "status": "start"})
    try:
        ir = extractor.repair(ir, raw_excerpt)
    except Exception as e:                                  # noqa: BLE001
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

    # ----- 5. Enhance per section (Summary, every Bullet, every Achievement) -----
    role_keywords = list_role_keywords(cfg.role_id)
    section_traces: List[SectionTrace] = []

    # Summary
    if ir.summary and not _looks_like_placeholder(ir.summary):
        _emit(progress, "section", {"label": "Summary", "status": "start"})
        trace = orchestrator.run(
            section_id="summary",
            label="Professional Summary",
            section_type="summary",
            original=ir.summary,
            role_id=cfg.role_id,
            priority_keywords=role_keywords,
        )
        if trace.changed:
            ir.summary = trace.after
        section_traces.append(trace)
        _emit(progress, "section", {
            "label": "Summary", "status": "done",
            "score": trace.final_score, "changed": trace.changed,
        })

    # Experience bullets
    for bi, blk in enumerate(ir.experience):
        if blk.placeholder:
            continue
        lead_bullets = set(plan.lead_bullet_hints.get(bi, []))
        for ji, bullet in enumerate(blk.bullets):
            if not bullet.strip():
                continue
            label = f"Bullet {bi+1}.{ji+1} - {blk.title or blk.company}"
            _emit(progress, "section", {"label": label, "status": "start"})
            trace = orchestrator.run(
                section_id=_bullet_id(bi, "bullet", ji),
                label=label,
                section_type="bullet",
                original=bullet,
                role_id=cfg.role_id,
                priority_keywords=role_keywords,
                is_lead_bullet=(ji in lead_bullets),
            )
            if trace.changed:
                blk.bullets[ji] = trace.after
            section_traces.append(trace)
            _emit(progress, "section", {
                "label": label, "status": "done",
                "score": trace.final_score, "changed": trace.changed,
            })
        for gi, group in enumerate(blk.groups):
            for ji, bullet in enumerate(group.bullets):
                if not bullet.strip():
                    continue
                label = f"Bullet {bi+1}.{gi+1}.{ji+1} - {group.label}"
                _emit(progress, "section", {"label": label, "status": "start"})
                trace = orchestrator.run(
                    section_id=_bullet_id(bi, "bullet", ji, gi),
                    label=label,
                    section_type="bullet",
                    original=bullet,
                    role_id=cfg.role_id,
                    priority_keywords=role_keywords,
                )
                if trace.changed:
                    group.bullets[ji] = trace.after
                section_traces.append(trace)
                _emit(progress, "section", {
                    "label": label, "status": "done",
                    "score": trace.final_score, "changed": trace.changed,
                })

    # Project bullets
    for pi, proj in enumerate(ir.projects):
        if proj.placeholder:
            continue
        for ji, bullet in enumerate(proj.bullets):
            if not bullet.strip():
                continue
            label = f"Project {pi+1}.{ji+1} - {proj.name}"
            trace = orchestrator.run(
                section_id=f"proj_{pi}_bullet_{ji}",
                label=label,
                section_type="project_bullet",
                original=bullet,
                role_id=cfg.role_id,
                priority_keywords=role_keywords,
            )
            if trace.changed:
                proj.bullets[ji] = trace.after
            section_traces.append(trace)

    # Achievements - rewrite the description only
    for ai, ach in enumerate(ir.achievements):
        if ach.placeholder or not ach.description.strip():
            continue
        label = f"Achievement {ai+1} - {ach.title}"
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
    _emit(progress, "stage", {
        "name": "render", "status": "done",
        "tex_path": str(out_tex),
        "char_count": len(tex),
    })

    # ----- 7. ATS scoring (current role keywords) -----
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

    # ----- 8. JD matching (target role; optional cross-role) -----
    if cfg.enable_jd_matching:
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

    # ----- 9. Role review (LLM, target role only by default) -----
    if cfg.enable_role_review:
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

    # ----- finalize -----
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    result.status = "complete" if not result.errors else (
        "partial" if result.tex_path else "error"
    )
    return result


def _looks_like_placeholder(s: str) -> bool:
    return s.strip().startswith("[") and s.strip().endswith("]") and len(s) < 200
