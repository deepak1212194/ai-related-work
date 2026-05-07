"""
enhancer.py — Section orchestration with the 2-agent loop + safety guards
============================================================================

Architecture (v3 — agentic):

    parsed sections
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Per section: IterativeOrchestrator                         │
    │     ┌──────────────┐         ┌──────────────┐               │
    │     │ EnhancerAgent│ ──draft─▶│ CriticAgent  │              │
    │     │ (skill-driven│         │ (rubric scorer│              │
    │     │  prompt)     │ ◀──fix──│  → JSON)     │              │
    │     └──────────────┘         └──────────────┘               │
    │       max 3 iterations · accept ≥ 80 · early-stop Δ < 3      │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼  best_text from orchestrator
    ┌─────────────────────────────────────────────────────────────┐
    │  Deterministic Python guards (UNCHANGED):                   │
    │     - length ratio ≥ 50%                                    │
    │     - protected-term drop check                             │
    │  → returns (chosen_text, was_replaced, reason)              │
    └─────────────────────────────────────────────────────────────┘

Hard guarantees preserved from v2:
  - Every LLM call is wrapped in try/except — pipeline never raises.
  - Bullet count is capped via `settings.max_bullets_to_enhance`.
  - Overall wall-clock is bounded; remaining sections kept verbatim.
  - Per-section output is length-checked AND keyword-checked.
  - Returns previews + iteration traces for the UI.

Sections WITH critic loop:    summary, bullet
Sections WITHOUT critic loop: skills (light polish), achievement (terse)
  — these are conservative polishes where iteration adds cost without lift.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable

from .agents import CriticAgent, EnhancerAgent, IterativeOrchestrator
from .config import settings
from .llm import LLMClient, LLMError
from .rules import (
    compose_system_prompt,
    get_summary_task,
    get_bullet_task,
    get_skills_task,
    get_achievement_task,
)
from .schemas import IterationStep, ParsedResume, SectionPreview

log = logging.getLogger(__name__)

# Tech terms / numbers we never want to lose silently
PROTECTED_TERMS_RE = re.compile(
    r"\b(LLM|RAG|FAISS|SBERT|CLIP|YOLO[v\d-]*|OC-?SORT|NvSORT|GPT-4o?|"
    r"Azure ML|Azure AI Search|AKS|Hugging Face|CrewAI|LangChain|"
    r"NVIDIA|DGX|NIM|TensorRT|ONNX|TRTexec|FastAPI|Kubernetes|Terraform|"
    r"Docker|Postgres(?:QL)?|Redis|Kafka|GraphQL|Tableau|Looker|"
    r"\d{2,}K\+?|\d+\.\d+|R\^?\d|MAE|RMSE|fine[- ]?tun\w+)",
    re.IGNORECASE,
)

# ATS keyword list — common terms recruiters search for
ATS_KEYWORDS = {
    "python", "java", "javascript", "typescript", "go", "rust", "sql",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
    "fastapi", "flask", "django", "react", "node.js",
    "postgresql", "mongodb", "redis", "kafka", "elasticsearch",
    "ci/cd", "git", "linux", "agile", "scrum",
    "llm", "rag", "nlp", "computer vision", "deep learning",
    "machine learning", "data pipeline", "etl", "api",
    "microservices", "distributed systems", "rest",
    "fine-tuning", "model training", "inference",
}


def _safe_replace(original: str, candidate: str, *,
                  min_ratio: float = 0.5) -> tuple[str, bool, str]:
    """
    Decide whether to use the LLM candidate or keep the input.
    Returns (text_to_use, was_replaced, reason_if_kept).
    """
    if not candidate or not candidate.strip():
        return original, False, "LLM returned empty"
    if len(candidate) < min_ratio * len(original):
        return original, False, f"output too short ({len(candidate)} < {int(min_ratio*100)}% of input)"
    original_terms = {m.group(0).lower() for m in PROTECTED_TERMS_RE.finditer(original)}
    candidate_terms = {m.group(0).lower() for m in PROTECTED_TERMS_RE.finditer(candidate)}
    dropped = original_terms - candidate_terms
    if dropped:
        return original, False, f"would drop protected terms: {', '.join(sorted(dropped))}"
    return candidate.strip(), True, ""


def _call_safe(llm: LLMClient, system: str, user: str) -> str:
    """Wrap LLM call. Returns "" on any failure — caller falls back."""
    try:
        return llm.complete(system, user)
    except LLMError as e:
        log.warning("[LLM] call failed: %s", e)
        return ""
    except Exception as e:                              # noqa: BLE001
        log.warning("[LLM] unexpected error: %s", e)
        return ""


# ──────────────────────────────────────────────────────────────────────
# ATS Keyword Scoring
# ──────────────────────────────────────────────────────────────────────
def compute_ats_score(text: str, role_keywords: list[str] | None = None) -> dict:
    """
    Compute ATS (Applicant Tracking System) keyword coverage score.
    """
    text_lower = text.lower()
    check_keywords = set(ATS_KEYWORDS)
    if role_keywords:
        check_keywords.update(k.lower() for k in role_keywords)

    matched, missing = [], []
    for kw in sorted(check_keywords):
        if kw in text_lower or kw.replace("-", " ") in text_lower or kw.replace(" ", "-") in text_lower:
            matched.append(kw)
        else:
            missing.append(kw)

    total = len(check_keywords)
    score = round(len(matched) / total * 100, 1) if total > 0 else 0

    suggestions = []
    if score < 30:
        suggestions.append("Very low keyword coverage — consider adding more technical terms")
    elif score < 50:
        suggestions.append("Below average coverage — add relevant technologies to Skills section")
    elif score < 70:
        suggestions.append("Good coverage — consider adding missing domain-specific terms")
    else:
        suggestions.append("Strong keyword coverage — well-optimized for ATS")

    high_impact = [k for k in missing if k in {
        "python", "docker", "kubernetes", "aws", "azure", "sql",
        "machine learning", "deep learning", "llm", "api", "ci/cd",
    }]
    if high_impact:
        suggestions.append(f"High-impact missing: {', '.join(high_impact[:5])}")

    return {
        "score": score,
        "matched_count": len(matched),
        "total_checked": total,
        "matched": matched[:20],
        "missing_high_impact": high_impact[:10],
        "suggestions": suggestions,
    }


# ──────────────────────────────────────────────────────────────────────
# Public — orchestrates everything with hard limits
# ──────────────────────────────────────────────────────────────────────
def enhance(parsed: ParsedResume, llm: LLMClient,
            role_id: str) -> tuple[ParsedResume, list[str], list[SectionPreview], list[str]]:
    """
    Returns (enhanced_parsed, notes, previews, warnings).

    Each preview includes the iteration trace from the agentic loop
    when applicable (summary + bullets). Skills + achievements use the
    legacy single-call path because the critic adds little there.
    """
    notes: list[str] = []
    previews: list[SectionPreview] = []
    warnings: list[str] = []
    started = time.monotonic()
    timeout = settings.overall_job_timeout_seconds

    def remaining_time() -> float:
        return max(0.0, timeout - (time.monotonic() - started))

    def time_up() -> bool:
        return remaining_time() <= 0.5

    system = compose_system_prompt(role_id)

    # Build the agent loop. Critic shares the same LLM client (one
    # backend, two roles). If the critic is disabled in settings, the
    # orchestrator degrades to single-call with no extra LLM hits.
    enhancer_agent = EnhancerAgent(llm, system)
    critic_agent = CriticAgent(llm) if settings.agent_critic_enabled else None
    orchestrator = IterativeOrchestrator(enhancer_agent, critic_agent)

    bullet_total = sum(len(b.bullets) for b in parsed.experience_blocks)
    cap = settings.max_bullets_to_enhance
    # Sections that get the critic loop (per settings)
    bullets_to_run = min(bullet_total, cap)
    sections_with_loop = (1 if parsed.summary else 0) + bullets_to_run
    # Per-section budget so the critic loop can never blow the global timeout.
    # 4× factor allows up to 4 LLM calls per section on average (1 draft +
    # critic + 1 retry-draft + critic) — most sections finish in 2.
    base_call = settings.llm_call_timeout_seconds
    per_section_budget = max(
        float(base_call) * 1.5,
        min(60.0, float(timeout) / max(1, sections_with_loop)),
    ) if sections_with_loop > 0 else float(base_call)

    use_loop_for = set(settings.agent_critique_sections)

    # ──────────────────────────────────────────────────────────────────
    # Summary  (agentic loop)
    # ──────────────────────────────────────────────────────────────────
    if parsed.summary and not time_up():
        before = parsed.summary
        task = get_summary_task(before)

        if "summary" in use_loop_for and critic_agent is not None:
            result = orchestrator.run(
                section_type="summary",
                original=before,
                task_prompt=task,
                time_budget_seconds=min(per_section_budget, remaining_time()),
            )
            cand = result.best_text
            iterations = result.iterations
            iters_used = result.used_iterations
            final_score = max((s.score for s in iterations), default=0.0)
        else:
            cand = _call_safe(llm, system, task)
            iterations, iters_used, final_score = [], 0, 0.0

        new_text, replaced, reason = _safe_replace(before, cand)
        parsed.summary = new_text
        previews.append(SectionPreview(
            section="Summary", before=before, after=new_text,
            changed=replaced, note="" if replaced else f"kept — {reason}",
            iterations=iterations,
            final_score=final_score,
            iterations_used=iters_used,
        ))
        notes.append(
            f"summary: {'enhanced' if replaced else 'kept'}"
            + (f" · {iters_used} iter(s) · score {int(final_score)}/100"
               if iters_used else "")
        )
    elif time_up():
        warnings.append("summary skipped — overall timeout reached")

    # ──────────────────────────────────────────────────────────────────
    # Skills  (legacy single-call: light polish, no critic)
    # ──────────────────────────────────────────────────────────────────
    skills_replaced = 0
    skills_processed = 0
    bucket_items = list(parsed.skills.items())[: settings.max_skills_buckets_to_enhance]
    if len(parsed.skills) > settings.max_skills_buckets_to_enhance:
        warnings.append(
            f"skills: {len(parsed.skills) - settings.max_skills_buckets_to_enhance}"
            f" buckets beyond cap — kept verbatim"
        )
    for bucket, items in bucket_items:
        if time_up():
            warnings.append(f"skills bucket '{bucket}' skipped — timeout")
            break
        skills_processed += 1
        task = get_skills_task(items)
        cand = _call_safe(llm, system, task)
        new_text, replaced, reason = _safe_replace(items, cand, min_ratio=0.7)
        parsed.skills[bucket] = new_text
        if replaced:
            skills_replaced += 1
    notes.append(f"skills: {skills_replaced}/{skills_processed} buckets enhanced")

    # ──────────────────────────────────────────────────────────────────
    # Experience bullets  (agentic loop)
    # ──────────────────────────────────────────────────────────────────
    bullets_replaced = 0
    bullets_processed = 0
    seen_bullets = 0
    bullet_iters_total = 0
    bullet_score_sum = 0.0
    bullet_score_n = 0

    for block in parsed.experience_blocks:
        new_bullets: list[str] = []
        for bullet in block.bullets:
            seen_bullets += 1
            if seen_bullets > cap or time_up():
                new_bullets.append(bullet)
                continue
            bullets_processed += 1
            task = get_bullet_task(bullet)

            if "bullet" in use_loop_for and critic_agent is not None:
                result = orchestrator.run(
                    section_type="bullet",
                    original=bullet,
                    task_prompt=task,
                    time_budget_seconds=min(per_section_budget, remaining_time()),
                )
                cand = result.best_text
                iterations = result.iterations
                iters_used = result.used_iterations
                final_score = max((s.score for s in iterations), default=0.0)
                bullet_iters_total += iters_used
                if final_score > 0:
                    bullet_score_sum += final_score
                    bullet_score_n += 1
            else:
                cand = _call_safe(llm, system, task)
                iterations, iters_used, final_score = [], 0, 0.0

            new_text, replaced, reason = _safe_replace(bullet, cand)
            new_bullets.append(new_text)
            previews.append(SectionPreview(
                section=f"Bullet · {block.title or 'experience'}",
                before=bullet, after=new_text,
                changed=replaced, note="" if replaced else f"kept — {reason}",
                iterations=iterations,
                final_score=final_score,
                iterations_used=iters_used,
            ))
            if replaced:
                bullets_replaced += 1
        block.bullets = new_bullets

    bullet_note = f"experience: {bullets_replaced}/{bullets_processed} bullets enhanced"
    if bullet_iters_total:
        avg_score = (
            bullet_score_sum / bullet_score_n if bullet_score_n else 0.0
        )
        bullet_note += (
            f" · {bullet_iters_total} total iter(s) · avg score {int(avg_score)}/100"
        )
    notes.append(bullet_note)

    if bullet_total > cap:
        warnings.append(
            f"experience: {bullet_total - cap} bullets beyond cap of {cap} kept verbatim"
        )
    if time_up():
        warnings.append("experience: overall timeout hit before all bullets processed")

    # ──────────────────────────────────────────────────────────────────
    # Achievements  (legacy single-call: terse polish, no critic)
    # ──────────────────────────────────────────────────────────────────
    ach_replaced = 0
    new_ach: list[str] = []
    for line in parsed.achievements:
        if time_up():
            new_ach.append(line)
            continue
        task = get_achievement_task(line)
        cand = _call_safe(llm, system, task)
        new_text, replaced, reason = _safe_replace(line, cand, min_ratio=0.7)
        new_ach.append(new_text)
        if replaced:
            ach_replaced += 1
    parsed.achievements = new_ach
    notes.append(f"achievements: {ach_replaced}/{len(new_ach)} lines polished")

    return parsed, notes, previews, warnings


def count_sections(parsed: ParsedResume) -> int:
    n = 0
    if parsed.summary: n += 1
    if parsed.skills: n += 1
    if parsed.experience_blocks: n += 1
    if parsed.education_blocks: n += 1
    if parsed.achievements: n += 1
    return n
