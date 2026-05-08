"""
JDMatchAgent - deterministic keyword scoring against curated job descriptions.

Loads `data/jds/<role_id>.json` and computes per-JD scores plus a roll-up
report. Synonyms are honoured: a match against any synonym in the bundle
counts as a match for the canonical keyword.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from ..core.config import JD_DIR
from ..core.ir import JDMatch, JDMatchReport

log = logging.getLogger(__name__)


@dataclass
class JDSample:
    title: str
    archetype: str
    seniority: str
    must_have: List[str]
    nice_to_have: List[str]
    synonyms: Dict[str, List[str]] = field(default_factory=dict)


def load_role_jds(role_id: str) -> List[JDSample]:
    p = JD_DIR / f"{role_id}.json"
    if not p.exists():
        log.warning("[JD] no JD file for role %s at %s", role_id, p)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                 # noqa: BLE001
        log.warning("[JD] failed to load %s: %s", p, e)
        return []
    out: List[JDSample] = []
    for s in data.get("samples", []):
        out.append(JDSample(
            title=str(s.get("title", "")),
            archetype=str(s.get("archetype", "")),
            seniority=str(s.get("seniority", "")),
            must_have=list(s.get("must_have", [])),
            nice_to_have=list(s.get("nice_to_have", [])),
            synonyms=dict(s.get("synonyms", {})),
        ))
    return out


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def _has_keyword(text_norm: str, keyword: str, synonyms: Sequence[str]) -> bool:
    candidates = [keyword] + list(synonyms)
    for c in candidates:
        c_norm = _normalize(c).strip()
        if not c_norm:
            continue
        if " " in c_norm:
            # Phrase: substring match (case-insensitive)
            if c_norm in text_norm:
                return True
        else:
            # Whole-word match for single tokens
            pattern = r"(?:^|[^a-z0-9+])" + re.escape(c_norm) + r"(?:[^a-z0-9+]|$)"
            if re.search(pattern, text_norm):
                return True
    return False


def _score_one(sample: JDSample, text_norm: str) -> tuple[float, int, int, List[str]]:
    must_total = len(sample.must_have)
    nice_total = len(sample.nice_to_have)
    matched_count = 0
    keywords_total = must_total * 2 + nice_total
    matched_pts = 0
    missing: List[str] = []
    for kw in sample.must_have:
        if _has_keyword(text_norm, kw, sample.synonyms.get(kw, [])):
            matched_count += 1
            matched_pts += 2
        else:
            missing.append(kw)
    for kw in sample.nice_to_have:
        if _has_keyword(text_norm, kw, sample.synonyms.get(kw, [])):
            matched_count += 1
            matched_pts += 1
    score = (matched_pts / keywords_total * 100.0) if keywords_total else 0.0
    return round(score, 1), matched_count, must_total + nice_total, missing


class JDMatchAgent:
    """Deterministic - does NOT use the LLM."""

    def evaluate(
        self,
        *,
        role_id: str,
        text_before: str,
        text_after: str,
    ) -> JDMatchReport:
        samples = load_role_jds(role_id)
        report = JDMatchReport(role_id=role_id, samples_count=len(samples))
        if not samples:
            return report
        norm_before = _normalize(text_before)
        norm_after = _normalize(text_after)
        gap_counter: Counter[str] = Counter()
        for s in samples:
            sb, mb, total, _miss_b = _score_one(s, norm_before)
            sa, ma, _t, miss_a = _score_one(s, norm_after)
            gap_counter.update(miss_a)
            report.samples.append(JDMatch(
                role_id=role_id,
                title=s.title,
                company_archetype=s.archetype,
                seniority=s.seniority,
                keywords_total=total,
                keywords_matched_before=mb,
                keywords_matched_after=ma,
                score_before=sb,
                score_after=sa,
                delta=round(sa - sb, 1),
                missing_keywords=miss_a[:10],
            ))
        if report.samples:
            report.avg_score_before = round(
                sum(s.score_before for s in report.samples) / len(report.samples), 1,
            )
            report.avg_score_after = round(
                sum(s.score_after for s in report.samples) / len(report.samples), 1,
            )
            report.avg_delta = round(report.avg_score_after - report.avg_score_before, 1)
        report.top_gaps = [kw for kw, _ in gap_counter.most_common(10)]
        return report
