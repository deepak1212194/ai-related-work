"""
RoleReviewerAgent - simulates a hiring manager's read of the full resume
for a given target role and returns a structured review.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.ir import RoleReview
from ..core.llm import LLMError
from ..core.skills import SkillBundle
from .base import Agent, coerce_score, extract_json

log = logging.getLogger(__name__)


class RoleReviewerAgent(Agent):
    def review(self, *, resume_text: str, role_id: str, role_name: str) -> RoleReview:
        # Trim resume text aggressively to keep prompt small
        text = resume_text[:8000]
        try:
            sys = (
                self.skills.get_block("00_core_rules")
                + "\n\n----\n\n"
                + self.skills.get_block("06_role_review")
                + "\n\n----\n\n"
                + self.skills.get_role(role_id)
                + "\n\nReturn JSON ONLY."
            )
            user = (
                f"TARGET_ROLE: {role_id}  ({role_name})\n\n"
                f"FULL_RESUME_TEXT:\n\"\"\"\n{text}\n\"\"\"\n\n"
                "Output the JSON now."
            )
            raw = self.llm.complete(sys, user, max_tokens=900, temperature=0.2)
        except LLMError as e:
            log.warning("[RoleReviewer:%s] LLM error: %s", role_id, e)
            return RoleReview(
                role_id=role_id, role_name=role_name,
                one_line_verdict=f"(unavailable: {e})",
            )
        except Exception as e:                              # noqa: BLE001
            log.warning("[RoleReviewer:%s] unexpected error: %s", role_id, e)
            return RoleReview(
                role_id=role_id, role_name=role_name,
                one_line_verdict=f"(unavailable: {e})",
            )
        data = extract_json(raw)
        if not isinstance(data, dict):
            return RoleReview(
                role_id=role_id, role_name=role_name,
                one_line_verdict="(critic returned unparseable JSON)",
            )
        return RoleReview(
            role_id=role_id,
            role_name=role_name,
            overall_score=coerce_score(data.get("overall_score")),
            strengths=[str(s)[:240] for s in (data.get("strengths") or [])[:8]],
            weaknesses=[str(s)[:240] for s in (data.get("weaknesses") or [])[:8]],
            missing_keywords=[str(s)[:64] for s in (data.get("missing_keywords") or [])[:12]],
            one_line_verdict=str(data.get("one_line_verdict") or "")[:240],
        )
