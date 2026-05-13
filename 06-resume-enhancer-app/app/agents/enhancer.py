"""
EnhancerAgent - rewrites a single section.

Returns plain-text only. Iteration support: if a previous draft + critique
is supplied, the agent gets a "fix this" message rather than a from-scratch
rewrite, focusing the model on the violations.

Now receives skills context so role keywords can be grounded against
the user's actual skills — prevents hallucinating keywords the user
doesn't have.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..core.context_budget import estimate_tokens, relevant_keywords
from ..core.llm import LLMError
from .base import Agent, clean_draft

# Parse "[N] text" lines from a block response
_BLOCK_IDX_RE = re.compile(r"^\[(\d+)\]\s+(.+)", re.MULTILINE)


def _parse_block_response(text: str, expected: int) -> list[str]:
    """Extract indexed bullets from a '[N] ...' formatted LLM response."""
    hits: dict[int, str] = {}
    for m in _BLOCK_IDX_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= expected:
            hits[idx] = m.group(2).strip()
    return [hits.get(i, "") for i in range(1, expected + 1)]

log = logging.getLogger(__name__)


SECTION_TYPES = {
    "summary", "bullet", "skills", "project_bullet", "achievement",
}


class EnhancerAgent(Agent):
    def system_prompt(self, role_id: str, section_type: str) -> str:
        compact_role = self.skills.get_role_blocks(
            role_id,
            ["priority_keywords", "hiring_signals", "red_flags"],
        ) or self.skills.get_role(role_id)
        return (
            "<core_rules>\n"
            + self.skills.get_block("00_core_rules")
            + "\n</core_rules>\n\n"
            + "<enhancement_task>\n"
            + self.skills.get_block("04_enhancement")
            + "\n</enhancement_task>\n\n"
            + "<role_profile>\n"
            + compact_role
            + "\n</role_profile>\n\n"
            + f"ACTIVE_SECTION_TYPE: {section_type}\n"
            + "STRICT_OUTPUT_CONTRACT: return exactly one plain-text rewrite; no markdown, no labels."
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
        skills_context: Optional[str] = None,
    ) -> str:
        if section_type not in SECTION_TYPES:
            section_type = "bullet"
        sys = self.system_prompt(role_id, section_type)
        selected_keywords = relevant_keywords(
            original=original,
            role_keywords=priority_keywords or [],
            skills_context=skills_context or "",
            max_items=14 if section_type in ("bullet", "project_bullet") else 18,
        )
        kw_block = ""
        if selected_keywords:
            kw_block = (
                "\nROLE_PRIORITY_KEYWORDS — these are confirmed in USER_SKILLS; "
                "weave applicable ones naturally into the rewrite:\n"
                + ", ".join(selected_keywords) + "\n"
            )
        skills_block = ""
        if skills_context:
            skills_block = (
                "\nUSER_SKILLS (authoritative source — you MAY name any skill listed here "
                "when it is relevant to this bullet's work):\n"
                + skills_context + "\n"
            )
        emphasis = ""
        if is_lead_bullet:
            emphasis = (
                "\nLEAD_BULLET: this is one of the highlight bullets for the block. "
                "Invest in a clear scope phrase and a specific outcome.\n"
            )
        # Use bounded output size to avoid verbose drift and speed up completion.
        max_tok = 420 if section_type == "summary" else 240
        user = (
            f"ORIGINAL ({section_type}):\n\"\"\"\n{original}\n\"\"\"\n"
            + kw_block + skills_block + emphasis +
            "\nRewrite rules:\n"
            "- Preserve every number, company name, technology, and duration from ORIGINAL.\n"
            "- You MAY add role keywords from ROLE_PRIORITY_KEYWORDS when they are listed in "
            "USER_SKILLS and naturally apply to this bullet's work — this is keyword surfacing, not fabrication.\n"
            "- Never invent metrics, outcomes, or tools absent from both ORIGINAL and USER_SKILLS.\n"
            "- Prefer tight prose; do not pad.\n"
            "\nReturn plain text only.\n"
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
        # If prompts got too large, trim optional skills context from the tail.
        if estimate_tokens(sys + user) > 2800 and skills_context:
            user = user.replace(skills_block, "")
        try:
            raw = self.llm.complete(sys, user, max_tokens=max_tok, temperature=0.25)
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

    def draft_block(
        self,
        *,
        section_type: str,
        block_context: str,
        bullets: list[str],
        role_id: str,
        priority_keywords: Optional[list[str]] = None,
        prior_drafts: Optional[list[str]] = None,
        prior_critique: Optional[dict] = None,
        lead_indices: Optional[set[int]] = None,
        skills_context: Optional[str] = None,
    ) -> list[str]:
        """Enhance all bullets from one experience/project block in a single LLM call.

        Returns a list of the same length as `bullets`.  An empty string at
        position i signals that the LLM did not produce a usable rewrite for
        that bullet — the caller should fall back to the original for that slot.
        Returns an empty list on total failure so the caller can fall back to
        per-bullet processing.
        """
        if not bullets:
            return []

        n = len(bullets)
        if section_type not in SECTION_TYPES:
            section_type = "bullet"

        sys = self.system_prompt(role_id, section_type)

        selected_keywords = relevant_keywords(
            original=" ".join(bullets),
            role_keywords=priority_keywords or [],
            skills_context=skills_context or "",
            max_items=18,
        )
        kw_block = ""
        if selected_keywords:
            kw_block = (
                "\nROLE_PRIORITY_KEYWORDS — confirmed in USER_SKILLS; "
                "weave applicable ones into the rewrites naturally:\n"
                + ", ".join(selected_keywords) + "\n"
            )
        skills_block = ""
        if skills_context:
            skills_block = (
                "\nUSER_SKILLS (authoritative — you MAY name any skill here "
                "when relevant to that bullet's work):\n"
                + skills_context + "\n"
            )
        lead_hint = ""
        if lead_indices:
            nums = ", ".join(f"[{i + 1}]" for i in sorted(lead_indices))
            lead_hint = (
                f"\nLEAD_BULLETS: {nums} — highlight bullets; "
                "invest in a clear scope phrase and a specific measured outcome.\n"
            )

        bullets_text = "\n".join(f"[{i + 1}] {b}" for i, b in enumerate(bullets))
        fmt_example  = "\n".join(f"[{i + 1}] <rewrite>" for i in range(n))

        user = (
            f"BLOCK_CONTEXT: {block_context}\n\n"
            f"ORIGINAL BULLETS:\n{bullets_text}\n"
            + kw_block + skills_block + lead_hint +
            f"\nRewrite ALL {n} bullets. Return EXACTLY {n} lines:\n"
            + fmt_example +
            "\n\nRules:\n"
            "- Same count as input — never merge, never split, never omit.\n"
            "- One line per bullet, prefixed [N] exactly as shown.\n"
            "- Preserve every number, company name, and technology exactly.\n"
            "- You MAY add keywords from ROLE_PRIORITY_KEYWORDS when they are in USER_SKILLS "
            "and naturally apply to that bullet's work — name the skill, not just the concept.\n"
            "- Never invent metrics, outcomes, or tools absent from both the bullet and USER_SKILLS.\n"
            "- No markdown, no labels, no explanations outside the [N] lines.\n"
        )

        if prior_drafts is not None and prior_critique is not None:
            prior_text = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(prior_drafts))
            violations = prior_critique.get("violations", [])
            fix_hint   = prior_critique.get("fix_hint", "")
            user += "\n----\n\nPREVIOUS DRAFTS (rejected):\n" + prior_text
            user += "\n\nCRITIC FEEDBACK:\n"
            for v in violations[:6]:
                user += f"- {v}\n"
            if fix_hint:
                user += f"\nFix hint: {fix_hint}\n"
            user += (
                f"\nProduce NEW drafts for all {n} bullets addressing every violation. "
                "Same [N] format. Plain text only.\n"
            )

        # Token budget scales with bullet count; cap at 2400
        max_tok = min(200 * n, 2400)

        if estimate_tokens(sys + user) > 3600 and skills_context:
            user = user.replace(skills_block, "")

        try:
            raw = self.llm.complete(sys, user, max_tokens=max_tok, temperature=0.25)
        except LLMError as e:
            log.warning("[Enhancer.block] LLM error: %s", e)
            return []
        except Exception as e:                              # noqa: BLE001
            log.warning("[Enhancer.block] unexpected error: %s", e)
            return []

        parsed  = _parse_block_response(raw, n)
        cleaned = [clean_draft(p, max_paragraphs=1) if p else "" for p in parsed]

        # If fewer than half the bullets came back, signal total failure
        non_empty = sum(1 for c in cleaned if c)
        if non_empty < max(1, n // 2):
            log.warning(
                "[Enhancer.block] only %d/%d bullets parsed from response — triggering fallback",
                non_empty, n,
            )
            return []

        return cleaned
