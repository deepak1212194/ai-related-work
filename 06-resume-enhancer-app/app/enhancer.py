"""
enhancer.py — Section-by-section orchestration with safety guards
==================================================================

Hard guarantees:
  - Every LLM call is wrapped in try/except. The app never raises.
  - Bullet count is capped; excess is preserved verbatim.
  - Overall wall-clock is bounded; if total elapsed exceeds timeout,
    remaining sections are kept as-is.
  - Per-section output is length-checked AND keyword-checked.
  - Returns previews for the UI with before/after + ATS score.

Now uses skill-file-driven task templates loaded from skills/*.md.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable

from .config import settings
from .llm import LLMClient, LLMError
from .rules import (
    compose_system_prompt,
    get_summary_task,
    get_bullet_task,
    get_skills_task,
    get_achievement_task,
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

    Returns:
        - score: 0-100 percentage
        - matched: keywords found
        - missing: important keywords not found
        - suggestions: improvement hints
    """
    text_lower = text.lower()

    # Base ATS keywords
    check_keywords = set(ATS_KEYWORDS)

    # Add role-specific keywords
    if role_keywords:
        check_keywords.update(k.lower() for k in role_keywords)

    matched = []
    missing = []

    for kw in sorted(check_keywords):
        # Check for the keyword or close variants
        if kw in text_lower or kw.replace("-", " ") in text_lower or kw.replace(" ", "-") in text_lower:
            matched.append(kw)
        else:
            missing.append(kw)

    total = len(check_keywords)
    score = round(len(matched) / total * 100, 1) if total > 0 else 0

    # Generate suggestions
    suggestions = []
    if score < 30:
        suggestions.append("Very low keyword coverage — consider adding more technical terms")
    elif score < 50:
        suggestions.append("Below average coverage — add relevant technologies to Skills section")
    elif score < 70:
        suggestions.append("Good coverage — consider adding missing domain-specific terms")
    else:
        suggestions.append("Strong keyword coverage — well-optimized for ATS")

    # Highlight top missing keywords (most impactful)
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
        "matched": matched[:20],  # Cap for response size
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
        task = get_summary_task(before)
        cand = _call_safe(llm, system, task)
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
        task = get_skills_task(items)
        cand = _call_safe(llm, system, task)
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
            task = get_bullet_task(bullet)
            cand = _call_safe(llm, system, task)
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
