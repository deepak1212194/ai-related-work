"""
CriticAgent - scores a draft against the original.

Returns a dict with `scores`, `total`, `violations`, `fix_hint`, `verdict`.
Falls back to a permissive accept on any LLM / parse error so the loop
keeps moving.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.llm import LLMError
from .base import Agent, coerce_score, extract_json

log = logging.getLogger(__name__)


class CriticAgent(Agent):
    def system_prompt(self) -> str:
        return (
            self.skills.get_block("00_core_rules")
            + "\n\n----\n\n"
            + self.skills.get_block("05_critique")
            + "\n\nReturn JSON ONLY. No prose, no markdown fences."
        )

    def score(self, section_type: str, before: str, after: str) -> dict:
        if not after.strip():
            return self._fallback("empty draft")
        try:
            sys = self.system_prompt()
            user = (
                f"SECTION_TYPE: {section_type}\n\n"
                f"ORIGINAL:\n\"\"\"\n{before}\n\"\"\"\n\n"
                f"REWRITE:\n\"\"\"\n{after}\n\"\"\"\n\n"
                "Output the JSON now."
            )
            raw = self.llm.complete(sys, user, max_tokens=500, temperature=0.0)
        except LLMError as e:
            log.warning("[Critic] LLM error: %s", e)
            return self._fallback("llm_error")
        except Exception as e:                              # noqa: BLE001
            log.warning("[Critic] unexpected error: %s", e)
            return self._fallback("exception")
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            return self._fallback("parse_error")
        raw_scores = parsed.get("scores") or {}
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        # Each per-dimension score is on a 0-20 scale; coerce permissively
        # so strings like "16/20", "16", or "0.8" all land as floats.
        dim_scores = {
            k: coerce_score(v, scale_to=20.0)
            for k, v in raw_scores.items()
        }
        total_raw = parsed.get("total")
        if total_raw is None:
            total = sum(dim_scores.values())
        else:
            total = coerce_score(total_raw, scale_to=100.0)
            # If the model echoed a 0-20 average instead of a 0-100 sum,
            # back off to the dimension sum which is the contract.
            if total < 20 and dim_scores and sum(dim_scores.values()) > total + 5:
                total = sum(dim_scores.values())
        violations = parsed.get("violations") or []
        if not isinstance(violations, list):
            violations = [str(violations)]
        verdict = parsed.get("verdict") or ("accept" if total >= 82 else "iterate")
        if verdict not in ("accept", "iterate"):
            verdict = "iterate"
        return {
            "scores": {k: int(v) for k, v in dim_scores.items()},
            "total": int(max(0, min(100, total))),
            "violations": [str(v)[:200] for v in violations[:6]],
            "fix_hint": str(parsed.get("fix_hint") or "")[:300],
            "verdict": verdict,
        }

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {
            "scores": {},
            "total": 80,
            "violations": [],
            "fix_hint": f"(critic unavailable: {reason})",
            "verdict": "accept",
        }
