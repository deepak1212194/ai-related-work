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
    """Select role keywords grounded in user evidence.

    Priority tiers:
      3 — keyword token(s) appear in BOTH the original bullet AND skills_context
      2 — keyword appears in skills_context only (user has the skill but hasn't
          mentioned it in this bullet — exactly the gap we want to fill)
      1 — keyword appears in the original bullet only (retention case)
      0 — keyword absent from both (not eligible: would require fabrication)
    """
    if not role_keywords:
        return []
    orig_terms   = extract_terms(original)
    skills_terms = extract_terms(skills_context)
    scored: List[tuple[int, str]] = []
    for kw in role_keywords:
        toks = [t for t in re.split(r"[^A-Za-z0-9+#./-]+", kw.lower()) if t]
        if not toks:
            continue
        in_orig   = sum(1 for t in toks if t in orig_terms)
        in_skills = sum(1 for t in toks if t in skills_terms)
        if in_orig > 0 and in_skills > 0:
            priority = 3
        elif in_skills > 0:
            priority = 2          # surface from skills into this bullet
        elif in_orig > 0:
            priority = 1          # keep what's already there
        else:
            continue              # genuinely absent from user's profile
        scored.append((priority, in_orig + in_skills, kw))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2].lower()))
    return normalize_terms([kw for _, _, kw in scored[:max_items]])

