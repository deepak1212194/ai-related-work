"""
EnhancerAgent - rewrites a single section.

Returns plain-text only. Iteration support: if a previous draft + critique
is supplied, the agent gets a "fix this" message rather than a from-scratch
rewrite, focusing the model on the violations.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.llm import LLMError
from .base import Agent, clean_draft

log = logging.getLogger(__name__)


SECTION_TYPES = {
    "summary", "bullet", "skills", "project_bullet", "achievement",
}


class EnhancerAgent(Agent):
    def system_prompt(self, role_id: str, section_type: str) -> str:
        return (
            self.skills.get_block("00_core_rules")
            + "\n\n----\n\n"
            + self.skills.get_block("04_enhancement")
            + "\n\n----\n\n"
            + self.skills.get_role(role_id)
            + f"\n\n----\n\nACTIVE_SECTION_TYPE: {section_type}"
        )

    def draft(
        self,
        *,
        section_type: str,
        original: str,
        role_id: str,
        priority_keywords: Optional[list[str]] = None,
        prior_draft: Optional[str] = None,
        prior_critique: Optional[dict] = None,
        is_lead_bullet: bool = False,
    ) -> str:
        if section_type not in SECTION_TYPES:
            section_type = "bullet"
        sys = self.system_prompt(role_id, section_type)
        kw_block = ""
        if priority_keywords:
            kw_block = (
                "\nROLE_PRIORITY_KEYWORDS (use only if they are TRUE for this user; never invent):\n"
                + ", ".join(priority_keywords[:30]) + "\n"
            )
        emphasis = ""
        if is_lead_bullet:
            emphasis = (
                "\nLEAD_BULLET: this is one of the highlight bullets for the block. "
                "Invest in a clear scope phrase and a specific outcome.\n"
            )
        user = (
            f"ORIGINAL ({section_type}):\n\"\"\"\n{original}\n\"\"\"\n"
            + kw_block + emphasis +
            "\nReturn the rewrite as plain text. No commentary.\n"
        )
        if prior_draft is not None and prior_critique is not None:
            violations = prior_critique.get("violations", [])
            fix_hint = prior_critique.get("fix_hint", "")
            user += "\n----\n\nPREVIOUS DRAFT (rejected):\n\"\"\"\n"
            user += prior_draft + "\n\"\"\"\n\nCRITIC FEEDBACK:\n"
            for v in violations[:6]:
                user += f"- {v}\n"
            if fix_hint:
                user += f"\nFix hint: {fix_hint}\n"
            user += (
                "\nProduce a NEW draft that addresses every violation. "
                "Keep all protected terms and numbers. Plain text only.\n"
            )
        try:
            raw = self.llm.complete(sys, user, max_tokens=420, temperature=0.25)
        except LLMError as e:
            log.warning("[Enhancer] LLM error: %s", e)
            return ""
        except Exception as e:                              # noqa: BLE001
            log.warning("[Enhancer] unexpected error: %s", e)
            return ""
        cleaned = clean_draft(raw, max_paragraphs=1)
        if not cleaned:
            log.debug("[Enhancer] cleanup stripped everything; raw=%r", raw[:200])
        return cleaned
