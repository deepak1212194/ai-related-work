"""
enhancer.py — Section-by-section orchestration
================================================

Applies the rules from rules.py to each parsed section via the LLM
backend. The orchestration is RESTRICTIVE by design:

  - Each section is rewritten independently (small, focused prompts).
  - The output is length-checked and length-clamped — if the LLM
    returns something suspiciously short (< 50% of input), we KEEP the
    input. This guards against accidental degradation.
  - Tech keyword preservation is checked per bullet; if a key noun
    disappears, we KEEP the input.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from .llm import LLMClient
from .rules import (
    ACHIEVEMENT_TASK,
    BULLET_TASK,
    SKILLS_TASK,
    SUMMARY_TASK,
    SYSTEM_RULES,
)
from .schemas import ParsedResume

log = logging.getLogger(__name__)

# Tech terms we never want to lose silently
PROTECTED_TERMS_RE = re.compile(
    r"\b(LLM|RAG|FAISS|SBERT|CLIP|YOLO[v\d-]*|OC-?SORT|NvSORT|GPT-4o?|"
    r"Azure ML|Azure AI Search|AKS|Hugging Face|CrewAI|LangChain|"
    r"NVIDIA|DGX|NIM|TensorRT|ONNX|TRTexec|FastAPI|"
    r"\d{2,}K\+?|\d+\.\d+|R\^?\d|MAE|RMSE|fine[- ]?tun\w+)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────
# Per-section enhancement with safety guards
# ──────────────────────────────────────────────────────────────────────
def _safe_replace(original: str, candidate: str, *,
                  min_ratio: float = 0.5) -> tuple[str, bool]:
    """
    Return (text_to_use, was_replaced).
    Reject the candidate if it's much shorter than the original or
    if it dropped a protected keyword that was present in the input.
    """
    if not candidate or candidate.strip() == "":
        return original, False
    if len(candidate) < min_ratio * len(original):
        log.warning("[GUARD] candidate too short — keeping original")
        return original, False

    original_terms = set(m.group(0).lower() for m in PROTECTED_TERMS_RE.finditer(original))
    candidate_terms = set(m.group(0).lower() for m in PROTECTED_TERMS_RE.finditer(candidate))
    dropped = original_terms - candidate_terms
    if dropped:
        log.warning("[GUARD] candidate dropped protected terms %s — keeping original", dropped)
        return original, False
    return candidate.strip(), True


def _call(llm: LLMClient, task: str) -> str:
    try:
        return llm.complete(SYSTEM_RULES, task)
    except Exception as e:                 # noqa: BLE001
        log.warning("[LLM] call failed: %s — falling back to input unchanged", e)
        return ""


# ──────────────────────────────────────────────────────────────────────
# Public: enhance a ParsedResume in place, return notes
# ──────────────────────────────────────────────────────────────────────
def enhance(parsed: ParsedResume, llm: LLMClient) -> tuple[ParsedResume, list[str]]:
    notes: list[str] = []

    # --- Summary ---
    if parsed.summary:
        cand = _call(llm, SUMMARY_TASK.format(content=parsed.summary))
        new_text, replaced = _safe_replace(parsed.summary, cand)
        parsed.summary = new_text
        notes.append(f"summary: {'enhanced' if replaced else 'kept (guard tripped)'}")

    # --- Skills ---
    skills_replaced = 0
    for bucket, items in parsed.skills.items():
        cand = _call(llm, SKILLS_TASK.format(content=items))
        new_text, replaced = _safe_replace(items, cand, min_ratio=0.7)
        parsed.skills[bucket] = new_text
        if replaced:
            skills_replaced += 1
    notes.append(f"skills: {skills_replaced}/{len(parsed.skills)} buckets enhanced")

    # --- Experience bullets ---
    bullets_replaced = 0
    bullets_total = 0
    for block in parsed.experience_blocks:
        new_bullets: list[str] = []
        for bullet in block.bullets:
            bullets_total += 1
            cand = _call(llm, BULLET_TASK.format(content=bullet))
            new_text, replaced = _safe_replace(bullet, cand)
            new_bullets.append(new_text)
            if replaced:
                bullets_replaced += 1
        block.bullets = new_bullets
    notes.append(f"experience: {bullets_replaced}/{bullets_total} bullets enhanced")

    # --- Achievements ---
    ach_replaced = 0
    new_ach: list[str] = []
    for line in parsed.achievements:
        cand = _call(llm, ACHIEVEMENT_TASK.format(content=line))
        new_text, replaced = _safe_replace(line, cand, min_ratio=0.7)
        new_ach.append(new_text)
        if replaced:
            ach_replaced += 1
    parsed.achievements = new_ach
    notes.append(f"achievements: {ach_replaced}/{len(new_ach)} lines polished")

    return parsed, notes


# ──────────────────────────────────────────────────────────────────────
# Convenience: how many sections were touched
# ──────────────────────────────────────────────────────────────────────
def count_sections(parsed: ParsedResume) -> int:
    n = 0
    if parsed.summary: n += 1
    if parsed.skills: n += 1
    if parsed.experience_blocks: n += 1
    if parsed.education_blocks: n += 1
    if parsed.achievements: n += 1
    return n
