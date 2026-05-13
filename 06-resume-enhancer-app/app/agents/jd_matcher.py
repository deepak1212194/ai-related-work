"""
JDMatchAgent - deterministic keyword scoring against curated job descriptions
and optionally a user-supplied custom JD.

Loads `data/jds/<role_id>.json` for the generic tab.
When a raw JD text is provided (custom tab), keywords are extracted from it
and scored against the resume — no curated file needed.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Custom JD keyword extraction (no curated file needed)
# ---------------------------------------------------------------------------

# Stop-words to strip from raw JD text before counting keyword candidates
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "on",
    "at", "by", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "must", "can", "not", "no", "but", "if", "as",
    "from", "that", "this", "these", "those", "we", "you", "they", "our",
    "your", "their", "its", "it", "i", "he", "she", "us", "them", "who",
    "which", "what", "when", "where", "how", "why", "more", "most", "also",
    "any", "all", "both", "about", "such", "than", "into", "through",
    "during", "before", "after", "above", "below", "between", "each",
    "other", "own", "same", "so", "just", "then", "too", "very", "well",
    "new", "good", "great", "strong", "preferred", "required", "desired",
    "experience", "ability", "knowledge", "understanding", "understanding",
    "including", "make", "work", "working", "ensure", "support", "use",
    "using", "across", "within", "across", "based", "including", "related",
    "including", "able", "key", "plus", "nice", "bonus",
}

# Technical multi-word phrases to recognise even without explicit listing
_TECH_BIGRAMS = re.compile(
    r"\b("
    r"machine learning|deep learning|natural language|computer vision|"
    r"large language|language model|neural network|reinforcement learning|"
    r"transfer learning|data pipeline|data science|data engineering|"
    r"feature engineering|model training|model serving|model deployment|"
    r"vector database|vector store|prompt engineering|retrieval augmented|"
    r"fine.?tuning|generative ai|agentic ai|multi.?agent|tool use|"
    r"a/b testing|distributed training|gpu cluster|inference server|"
    r"real.?time|end.?to.?end|full.?stack|open.?source"
    r")\b",
    re.IGNORECASE,
)


def extract_keywords_from_jd(jd_text: str) -> tuple[List[str], List[str]]:
    """Extract must-have and nice-to-have keywords from raw JD text.

    Heuristic rules:
    - Lines/phrases after "required", "must have", "you will need" → must_have
    - Lines/phrases after "preferred", "nice to have", "bonus", "plus" → nice_to_have
    - Capitalised acronyms (2-8 chars) and known tech phrases always → must_have
    - Remaining non-stop-word tokens from bullet points → nice_to_have
    """
    must: List[str] = []
    nice: List[str] = []
    seen: set[str] = set()

    def _add(lst: List[str], kw: str) -> None:
        k = kw.strip().lower()
        if k and k not in seen and len(kw) > 1:
            seen.add(k)
            lst.append(kw.strip())

    # Step 1 — extract multi-word tech phrases
    for m in _TECH_BIGRAMS.finditer(jd_text):
        _add(must, m.group(0))

    # Step 2 — collect capitalised acronyms (e.g. RAG, LLM, FAISS, CI/CD)
    for token in re.findall(r"\b[A-Z][A-Z0-9+#/.-]{1,7}\b", jd_text):
        if token not in {"AND", "OR", "THE", "FOR", "WITH", "NOT", "ARE",
                         "HAS", "HAVE", "CAN", "WILL", "MUST", "MAY", "YOUR",
                         "OUR", "ALL", "BE", "US", "AT", "IN", "TO", "OF",
                         "IS", "IT", "WE", "YOU", "NO", "DO", "AN", "BY"}:
            _add(must, token)

    # Step 3 — scan line context for required vs preferred signals
    nice_section = False
    for line in jd_text.splitlines():
        low = line.lower().strip()
        # Flip to nice_to_have section when preamble says so
        if re.search(r"\b(preferred|nice.to.have|bonus|plus|desired)\b", low):
            nice_section = True
        elif re.search(r"\b(required|must.have|you.will|you.must|essential|mandatory)\b", low):
            nice_section = False

        # Extract CamelCase / PascalCase tokens (library/framework names)
        for token in re.findall(r"\b[A-Z][a-z]{2,}(?:[A-Z][a-z]+)+\b", line):
            _add(must if not nice_section else nice, token)

        # Extract bracketed/version tokens: Python 3.x, PyTorch 2.x, etc.
        for token in re.findall(r"\b([A-Za-z][A-Za-z0-9+#._-]{2,15})\s+\d+[\.\d]*\b", line):
            _add(must if not nice_section else nice, token)

        # Bullet-point lines: extract meaningful nouns/terms
        if re.match(r"^\s*[-•*·‣▸▹◦]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line):
            _GENERIC = {
                "deep", "solid", "proven", "years", "expertise",
                "familiar", "familiarity", "proficiency", "proficient",
                "excellent", "hands", "preferred", "required",
                "following", "various", "similar",
            }
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#._-]{2,25}\b", line)
            for w in words:
                wl = w.lower()
                if wl in _STOP_WORDS or wl in _GENERIC or w.isdigit():
                    continue
                _add(nice if nice_section else must, w)

    # De-duplicate: anything in must shouldn't also be in nice
    must_lower = {m.lower() for m in must}
    nice = [n for n in nice if n.lower() not in must_lower]

    return must[:40], nice[:30]


def _score_custom_jd(
    jd_text: str,
    text_before: str,
    text_after: str,
) -> JDMatchReport:
    """Score a resume against a user-supplied raw JD text."""
    must_have, nice_to_have = extract_keywords_from_jd(jd_text)
    if not must_have and not nice_to_have:
        return JDMatchReport(role_id="custom", samples_count=0)

    dummy = JDSample(
        title="Your Target Job",
        archetype="Custom JD",
        seniority="",
        must_have=must_have,
        nice_to_have=nice_to_have,
    )
    norm_before = _normalize(text_before)
    norm_after = _normalize(text_after)
    sb, mb, total, _ = _score_one(dummy, norm_before)
    sa, ma, _, miss_a = _score_one(dummy, norm_after)

    report = JDMatchReport(role_id="custom", samples_count=1)
    report.samples.append(JDMatch(
        role_id="custom",
        title="Your Target Job",
        company_archetype="Custom JD",
        seniority="",
        keywords_total=total,
        keywords_matched_before=mb,
        keywords_matched_after=ma,
        score_before=sb,
        score_after=sa,
        delta=round(sa - sb, 1),
        missing_keywords=miss_a[:15],
    ))
    report.avg_score_before = sb
    report.avg_score_after = sa
    report.avg_delta = round(sa - sb, 1)
    report.top_gaps = miss_a[:10]
    return report


class JDMatchAgent:
    """Deterministic - does NOT use the LLM.

    Two evaluation paths:
    - evaluate()        → scores against curated role JD samples (generic tab)
    - evaluate_custom() → scores against a raw user-supplied JD text (custom tab)
    """

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

    def evaluate_custom(
        self,
        *,
        jd_text: str,
        text_before: str,
        text_after: str,
    ) -> JDMatchReport:
        """Score resume against a raw job description pasted by the user."""
        return _score_custom_jd(jd_text, text_before, text_after)

    def get_custom_keywords(self, jd_text: str) -> tuple[List[str], List[str]]:
        """Extract must-have and nice-to-have keywords from a raw JD for use
        as priority_keywords in the enhancer when custom JD mode is active."""
        return extract_keywords_from_jd(jd_text)
