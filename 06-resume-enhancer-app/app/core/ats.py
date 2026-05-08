"""
ats.py - keyword coverage scoring.

Lightweight, deterministic ATS-style score. Given a resume text and a
list of role keywords, returns a score, the matched set, and the missing
high-impact terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class ATSResult:
    score: float
    matched_count: int
    total_checked: int
    matched: List[str]
    missing: List[str]
    suggestions: List[str]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9+/.\- ]+", " ", s.lower())


def score_keywords(text: str, keywords: Sequence[str]) -> ATSResult:
    if not keywords:
        return ATSResult(0.0, 0, 0, [], [], [])
    norm = _normalize(text)
    matched: List[str] = []
    missing: List[str] = []
    for kw in keywords:
        kw_norm = _normalize(kw).strip()
        if not kw_norm:
            continue
        # Whole-token match where possible; fall back to substring for
        # multi-word phrases.
        pattern = (
            r"(?:^|[^a-z0-9])"
            + re.escape(kw_norm)
            + r"(?:[^a-z0-9]|$)"
        )
        if re.search(pattern, norm):
            matched.append(kw)
        else:
            missing.append(kw)
    total = len(matched) + len(missing)
    score = (len(matched) / total * 100.0) if total else 0.0
    suggestions: List[str] = []
    if missing:
        # Surface only the top 8 missing keywords as concrete suggestions.
        suggestions.append(
            "Consider weaving these high-signal terms into your resume "
            "where they truthfully describe your work: "
            + ", ".join(missing[:8]) + "."
        )
    return ATSResult(
        score=round(score, 1),
        matched_count=len(matched),
        total_checked=total,
        matched=matched,
        missing=missing[:30],
        suggestions=suggestions,
    )


def merge(*reports: ATSResult) -> ATSResult:
    """Merge multiple ATS reports (e.g. cross-role) into one summary."""
    if not reports:
        return ATSResult(0.0, 0, 0, [], [], [])
    matched: set[str] = set()
    missing: set[str] = set()
    total_checked = 0
    matched_count = 0
    for r in reports:
        matched.update(r.matched)
        missing.update(r.missing)
        total_checked += r.total_checked
        matched_count += r.matched_count
    score = (matched_count / total_checked * 100.0) if total_checked else 0.0
    return ATSResult(
        score=round(score, 1),
        matched_count=matched_count,
        total_checked=total_checked,
        matched=sorted(matched),
        missing=sorted(missing)[:30],
        suggestions=[],
    )
