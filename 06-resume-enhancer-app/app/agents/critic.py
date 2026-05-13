"""
CriticAgent - scores a draft against the original.

Returns a dict with `scores`, `total`, `violations`, `fix_hint`, `verdict`.
Falls back to a conservative "iterate" on LLM / parse errors so the loop
tries again rather than accepting a potentially bad draft.
On the FINAL iteration, falls back to "accept" to prevent infinite loops.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.config import settings
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

    def score(
        self,
        section_type: str,
        before: str,
        after: str,
        *,
        is_final_iteration: bool = False,
        priority_keywords: list[str] | None = None,
        skills_context: str | None = None,
    ) -> dict:
        if not after.strip():
            return self._fallback("empty draft", is_final=is_final_iteration)
        try:
            sys = self.system_prompt()
            kw_block = ""
            if priority_keywords:
                kw_block = (
                    f"\nROLE_PRIORITY_KEYWORDS: {', '.join(priority_keywords[:20])}\n"
                )
            skills_block = ""
            if skills_context:
                skills_block = f"\nUSER_SKILLS:\n{skills_context[:600]}\n"
            user = (
                f"SECTION_TYPE: {section_type}\n"
                + kw_block + skills_block +
                f"\nORIGINAL:\n\"\"\"\n{before}\n\"\"\"\n\n"
                f"REWRITE:\n\"\"\"\n{after}\n\"\"\"\n\n"
                "Output the JSON now."
            )
            raw = self.llm.complete(sys, user, max_tokens=500, temperature=0.0)
        except LLMError as e:
            log.warning("[Critic] LLM error: %s", e)
            return self._fallback("llm_error", is_final=is_final_iteration)
        except Exception as e:                              # noqa: BLE001
            log.warning("[Critic] unexpected error: %s", e)
            return self._fallback("exception", is_final=is_final_iteration)
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            return self._fallback("parse_error", is_final=is_final_iteration)
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
            # Normalise raw sum to 0-100: n_dims × 20 = max raw
            n_dims = max(len(dim_scores), 1)
            raw_sum = sum(dim_scores.values())
            total = round(raw_sum / (n_dims * 20) * 100)
        else:
            total = coerce_score(total_raw, scale_to=100.0)
            # If the model returned a 0-120 raw sum instead of normalised 0-100,
            # normalise it ourselves.
            if total > 100:
                n_dims = max(len(dim_scores), 6)
                total = round(total / (n_dims * 20) * 100)
            elif total < 20 and dim_scores and sum(dim_scores.values()) > total + 5:
                # Model echoed an average-per-dim; recompute from dimensions.
                n_dims = max(len(dim_scores), 1)
                raw_sum = sum(dim_scores.values())
                total = round(raw_sum / (n_dims * 20) * 100)
        violations = parsed.get("violations") or []
        if not isinstance(violations, list):
            violations = [str(violations)]
        verdict = parsed.get("verdict") or ("accept" if total >= settings.accept_threshold else "iterate")
        if verdict not in ("accept", "iterate"):
            verdict = "iterate"
        return {
            "scores": {k: int(v) for k, v in dim_scores.items()},
            "total": int(max(0, min(100, total))),
            "violations": [str(v)[:200] for v in violations[:6]],
            "fix_hint": str(parsed.get("fix_hint") or "")[:300],
            "verdict": verdict,
        }

    def score_block(
        self,
        section_type: str,
        originals: list[str],
        drafts: list[str],
        *,
        is_final_iteration: bool = False,
        priority_keywords: list[str] | None = None,
        skills_context: str | None = None,
    ) -> dict:
        """Score a batch of bullet rewrites as a single block.

        Produces one aggregate verdict (same schema as `score()`) that the
        orchestrator uses for the accept/iterate decision across the whole block.
        Delegates to `score()` when the block contains only one bullet.
        """
        if len(originals) == 1:
            return self.score(
                section_type, originals[0], drafts[0],
                is_final_iteration=is_final_iteration,
                priority_keywords=priority_keywords,
                skills_context=skills_context,
            )

        non_empty = [(o, d) for o, d in zip(originals, drafts) if d.strip()]
        if not non_empty:
            return self._fallback("all drafts empty", is_final=is_final_iteration)

        pairs = "\n\n".join(
            f"--- Bullet {i + 1} ---\nOriginal: {o}\nRewrite:  {d}"
            for i, (o, d) in enumerate(non_empty)
        )
        kw_block = ""
        if priority_keywords:
            kw_block = f"\nROLE_PRIORITY_KEYWORDS: {', '.join(priority_keywords[:20])}\n"
        skills_block = ""
        if skills_context:
            skills_block = f"\nUSER_SKILLS:\n{skills_context[:600]}\n"
        try:
            sys  = self.system_prompt()
            user = (
                f"SECTION_TYPE: {section_type}\n"
                + kw_block + skills_block +
                "\nScore the following block of bullet rewrites as a single unit. "
                "Return one JSON verdict covering the whole block.\n\n"
                + pairs +
                "\n\nOutput the JSON now."
            )
            raw = self.llm.complete(sys, user, max_tokens=600, temperature=0.0)
        except LLMError as e:
            log.warning("[Critic.block] LLM error: %s", e)
            return self._fallback("llm_error", is_final=is_final_iteration)
        except Exception as e:                              # noqa: BLE001
            log.warning("[Critic.block] unexpected error: %s", e)
            return self._fallback("exception", is_final=is_final_iteration)

        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            return self._fallback("parse_error", is_final=is_final_iteration)

        raw_scores = parsed.get("scores") or {}
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        dim_scores = {k: coerce_score(v, scale_to=20.0) for k, v in raw_scores.items()}
        total_raw  = parsed.get("total")
        if total_raw is None:
            n_dims = max(len(dim_scores), 1)
            raw_sum = sum(dim_scores.values())
            total = round(raw_sum / (n_dims * 20) * 100)
        else:
            total = coerce_score(total_raw, scale_to=100.0)
            if total > 100:
                n_dims = max(len(dim_scores), 6)
                total = round(total / (n_dims * 20) * 100)
            elif total < 20 and dim_scores and sum(dim_scores.values()) > total + 5:
                n_dims = max(len(dim_scores), 1)
                raw_sum = sum(dim_scores.values())
                total = round(raw_sum / (n_dims * 20) * 100)
        violations = parsed.get("violations") or []
        if not isinstance(violations, list):
            violations = [str(violations)]
        verdict = parsed.get("verdict") or ("accept" if total >= settings.accept_threshold else "iterate")
        if verdict not in ("accept", "iterate"):
            verdict = "iterate"
        return {
            "scores":     {k: int(v) for k, v in dim_scores.items()},
            "total":      int(max(0, min(100, total))),
            "violations": [str(v)[:200] for v in violations[:6]],
            "fix_hint":   str(parsed.get("fix_hint") or "")[:300],
            "verdict":    verdict,
        }

    @staticmethod
    def _fallback(reason: str, *, is_final: bool = False) -> dict:
        """Conservative fallback when the critic can't evaluate.

        On non-final iterations: returns "iterate" so the enhancer gets
        another chance rather than accepting an unreviewed draft.
        On the final iteration: returns "accept" to prevent infinite loops.
        """
        if is_final:
            return {
                "scores": {},
                "total": 75,
                "violations": [],
                "fix_hint": f"(critic unavailable on final iteration: {reason})",
                "verdict": "accept",
            }
        return {
            "scores": {},
            "total": 50,
            "violations": [f"critic unavailable: {reason}"],
            "fix_hint": f"(critic unavailable: {reason} — will retry)",
            "verdict": "iterate",
        }
