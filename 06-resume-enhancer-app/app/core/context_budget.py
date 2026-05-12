"""
context_budget.py - strict prompt/context token budgeting helpers.

Goal: keep prompts compact without sacrificing fidelity.
"""

from __future__ import annotations

import re
from typing import Iterable, List


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-/#]*")


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate for budget gating."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_terms(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        t = re.sub(r"\s+", " ", (v or "").strip())
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def extract_terms(text: str, *, min_len: int = 3) -> set[str]:
    out: set[str] = set()
    for m in _WORD_RE.finditer(text or ""):
        tok = m.group(0).strip().lower()
        if len(tok) >= min_len:
            out.add(tok)
    return out


def relevant_keywords(
    original: str,
    role_keywords: List[str],
    skills_context: str = "",
    *,
    max_items: int = 16,
) -> List[str]:
    """Select role keywords that overlap with user evidence."""
    if not role_keywords:
        return []
    evidence = extract_terms(original) | extract_terms(skills_context)
    scored: List[tuple[int, str]] = []
    for kw in role_keywords:
        toks = [t for t in re.split(r"[^A-Za-z0-9+#./-]+", kw.lower()) if t]
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in evidence)
        if overlap > 0:
            scored.append((overlap, kw))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    return normalize_terms([kw for _, kw in scored[:max_items]])

