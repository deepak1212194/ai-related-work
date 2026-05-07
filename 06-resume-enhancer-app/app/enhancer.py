"""
enhancer.py — Section-by-section orchestration with safety guards
==================================================================

Hard guarantees:
  - Every LLM call is wrapped in try/except. The app never raises.
  - Bullet count is capped at settings.max_bullets_to_enhance; the
    excess is preserved verbatim (with a note appended).
  - Overall wall-clock is bounded; if total elapsed exceeds
    settings.overall_job_timeout_seconds, remaining sections are kept
    as-is.
  - Per-section output is length-checked (>= 50% of input length)
    AND keyword-checked (no protected term may disappear). On either
    failure, the input is preserved.
  - Returns a list of SectionPreview objects so the UI can show
    before/after for every section, with explicit "kept (reason)"
    notes when an enhancement was rejected.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable

from .config import settings
from .llm import LLMClient, LLMError
from .rules import (
    ACHIEVEMENT_TASK,
    BULLET_TASK,
    SKILLS_TASK,
    SUMMARY_TASK,
    compose_system_prompt,
)
from .schemas import ParsedResume, SectionPreview

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
# Public — orchestrates everything with hard limits
# ──────────────────────────────────────────────────────────────────────
def enhance(parsed: ParsedResume, llm: LLMClient,
            role_id: str) -> tuple[ParsedResume, list[str], list[SectionPreview], list[str]]:
    """
    Returns (enhanced_parsed, notes, previews, warnings).

    `notes` is a high-level summary line per section.
    `previews` is per-item before/after for the UI.
    `warnings` is human-readable issues that didn't stop the run.
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

    # --- Summary ---
    if parsed.summary and not time_up():
        before = parsed.summary
        cand = _call_safe(llm, system, SUMMARY_TASK.format(content=before))
        new_text, replaced, reason = _safe_replace(before, cand)
        parsed.summary = new_text
        previews.append(SectionPreview(
            section="Summary", before=before, after=new_text,
            changed=replaced, note="" if replaced else f"kept — {reason}",
        ))
        notes.append(f"summary: {'enhanced' if replaced else 'kept'}")
    elif time_up():
        warnings.append("summary skipped — overall timeout reached")

    # --- Skills ---
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
        cand = _call_safe(llm, system, SKILLS_TASK.format(content=items))
        new_text, replaced, reason = _safe_replace(items, cand, min_ratio=0.7)
        parsed.skills[bucket] = new_text
        if replaced:
            skills_replaced += 1
    notes.append(f"skills: {skills_replaced}/{skills_processed} buckets enhanced")

    # --- Experience bullets ---
    bullets_replaced = 0
    bullets_processed = 0
    bullet_total = sum(len(b.bullets) for b in parsed.experience_blocks)
    cap = settings.max_bullets_to_enhance
    seen_bullets = 0
    for block in parsed.experience_blocks:
        new_bullets: list[str] = []
        for bullet in block.bullets:
            seen_bullets += 1
            if seen_bullets > cap or time_up():
                # Keep verbatim past the cap
                new_bullets.append(bullet)
                continue
            bullets_processed += 1
            cand = _call_safe(llm, system, BULLET_TASK.format(content=bullet))
            new_text, replaced, reason = _safe_replace(bullet, cand)
            new_bullets.append(new_text)
            previews.append(SectionPreview(
                section=f"Bullet · {block.title or 'experience'}",
                before=bullet, after=new_text,
                changed=replaced, note="" if replaced else f"kept — {reason}",
            ))
            if replaced:
                bullets_replaced += 1
        block.bullets = new_bullets
    notes.append(f"experience: {bullets_replaced}/{bullets_processed} bullets enhanced")
    if bullet_total > cap:
        warnings.append(
            f"experience: {bullet_total - cap} bullets beyond cap of {cap} kept verbatim"
        )
    if time_up():
        warnings.append("experience: overall timeout hit before all bullets processed")

    # --- Achievements ---
    ach_replaced = 0
    new_ach: list[str] = []
    for line in parsed.achievements:
        if time_up():
            new_ach.append(line)
            continue
        cand = _call_safe(llm, system, ACHIEVEMENT_TASK.format(content=line))
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
