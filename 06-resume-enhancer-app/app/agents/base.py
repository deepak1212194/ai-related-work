"""
base.py - shared Agent helpers.

Every agent inherits from `Agent` and gets:
- a typed `LLMClient` reference,
- a SkillBundle for hot-reloadable .md instructions,
- helpers `clean_draft()` and `extract_json()` for robust output parsing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..core.llm import LLMClient
from ..core.skills import SkillBundle, get_bundle

log = logging.getLogger(__name__)


@dataclass
class Agent:
    llm: LLMClient
    skills: SkillBundle

    @classmethod
    def with_default_skills(cls, llm: LLMClient) -> "Agent":
        return cls(llm=llm, skills=get_bundle())


# ----------------------------------------------------------------------
# Output cleanup helpers
# ----------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n?|\n?```\s*$", re.MULTILINE)
_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"here(?:'s| is)(?: the| a)? (?:rewritten|enhanced|polished|improved)?\s*"
    r"(?:bullet|summary|version|line|text|output|answer)[:\s]*|"
    r"(?:enhanced|rewritten|polished|improved)\s*"
    r"(?:bullet|summary|version|line|text)\s*[:\-]\s*|"
    r"output[:\-]\s*|"
    r"final(?:\s+(?:bullet|version|answer))?\s*[:\-]\s*|"
    r"sure[,!.]?\s*here.*?:\s*"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_NOISE_RE = re.compile(
    r"(?:\n\s*\n|\s*)("
    r"\(\s*note\s*:[\s\S]*$|"
    r"(?<=\n)note\s*:[\s\S]*$|"
    r"\(\s*explanation[\s\S]*$|"
    r"(?<=\n)explanation\s*:[\s\S]*$|"
    r"(?<=\n)rationale\s*:[\s\S]*$|"
    r"(?<=\n)iteration\s+\d[\s\S]*$|"
    r"this (?:bullet|output|summary|version|rewrite) (?:passes|scores|achieves)[\s\S]*$|"
    r"(?<=\n)differentiation[\s\S]*$"
    r")",
    re.IGNORECASE,
)


def clean_draft(text: str, *, max_paragraphs: int = 1) -> str:
    """Aggressively scrub LLM commentary out of a draft."""
    if not text:
        return ""
    s = text.strip()
    s = _FENCE_RE.sub("", s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    pre = _PREFIX_RE.match(s)
    if pre:
        s = s[pre.end():].strip()
    cut = _TRAILING_NOISE_RE.search(s)
    if cut:
        s = s[:cut.start()].rstrip()
    if max_paragraphs == 1:
        s = re.split(r"\n\s*\n", s, maxsplit=1)[0].strip()
    s = re.sub(r"^[•\-▪▸*–—o]\s+", "", s).strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\*+\s*$", "", s).strip()
    return s


def coerce_score(value: Any, *, scale_to: float = 100.0) -> float:
    """
    Coerce a permissive LLM-emitted score into a 0..scale_to float.

    Handles: int, float, "82", "82/100", "8/10", "82%", "82 %", and stray
    whitespace. Anything unparseable returns 0.0.

    Special care: a raw integer like 1, 2, 3 on a 0-100 scale is treated
    as literal (not expanded). Only values strictly between 0 and 1
    (exclusive) on a >=10 scale are treated as fractions and expanded.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        n = float(value)
        # Only expand true fractions (0 < n < 1.0), not integers like 1
        if 0.0 < n < 1.0 and scale_to >= 10 and not isinstance(value, int):
            return round(n * scale_to, 1)
        return float(min(max(n, 0.0), scale_to))
    s = str(value).strip()
    if not s:
        return 0.0
    s = s.rstrip("%").strip()
    # X/Y form -> normalise to scale_to
    if "/" in s:
        num, _, den = s.partition("/")
        try:
            n = float(num.strip())
            d = float(den.strip())
            if d > 0:
                return round((n / d) * scale_to, 1)
        except ValueError:
            return 0.0
    try:
        n = float(s)
    except ValueError:
        # Strip non-numeric junk and retry once
        digits = re.sub(r"[^0-9.\-]", "", s)
        try:
            n = float(digits) if digits else 0.0
        except ValueError:
            return 0.0
    # For string inputs, only expand true fractions
    if 0.0 < n < 1.0 and scale_to >= 10:
        return round(n * scale_to, 1)
    return float(min(max(n, 0.0), scale_to))


def extract_json(text: str) -> Optional[Any]:
    """Robust JSON extraction from a text body that may include prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Greedy first-brace match
    m = re.search(r"[\{\[][\s\S]*[\}\]]", text)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            # Try chopping any trailing junk after the last closing brace
            last_close = max(snippet.rfind("}"), snippet.rfind("]"))
            if last_close > 0:
                try:
                    return json.loads(snippet[: last_close + 1])
                except Exception:
                    pass
    return None
