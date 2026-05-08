"""
ExtractorAgent - LLM-assisted post-processing of the heuristic .tex parse.

The .tex parser does the heavy lifting deterministically; this agent's job
is to (a) repair common parse errors, (b) detect missed sections, and
(c) lift a headline from the most recent role when the input has none.

Falls back to the heuristic parse on any LLM failure.
"""

from __future__ import annotations

import logging
from typing import List

from ..core.ir import (
    ContactLink, ExperienceBlock, ExperienceGroup, ExtraSection, ResumeIR,
    SkillBucket,
)
from .base import Agent, extract_json

log = logging.getLogger(__name__)


class ExtractorAgent(Agent):
    def repair(self, ir: ResumeIR, raw_tex_excerpt: str) -> ResumeIR:
        """
        Best-effort LLM-assisted repair of the heuristic parse.

        Only used to fix obvious damage (e.g. degree -> location swap,
        empty headline). On any error, returns the input unchanged.
        """
        # Only call LLM if there's a chance of useful repair.
        needs_repair = (
            not ir.header.headline
            or any(not e.title and e.bullets for e in ir.experience)
            or any(not ed.institution for ed in ir.education)
        )
        if not needs_repair:
            return ir
        try:
            sys_prompt = (
                self.skills.get_block("00_core_rules")
                + "\n\n----\n\n"
                + self.skills.get_block("01_extraction")
                + "\n\nReturn ONLY a JSON object with the fields listed in the schema_hints. "
                "Do not add commentary."
            )
            user = (
                f"CURRENT_PARSE (JSON):\n{ir.model_dump_json()}\n\n"
                f"RAW_TEX_FIRST_3KB:\n```\n{raw_tex_excerpt[:3000]}\n```\n\n"
                "Return a JSON object with these top-level keys (only include those you want to change):\n"
                "- headline (string)\n"
                "- experience_repairs: list of {index, title?, company?, dates?}\n"
                "- education_repairs: list of {index, institution?, location?}\n"
                "- missed_sections: list of {title, items: [string, ...]}\n"
            )
            raw = self.llm.complete(sys_prompt, user, max_tokens=900, temperature=0.1)
        except Exception as e:                              # noqa: BLE001
            log.warning("[Extractor] LLM repair failed: %s", e)
            return ir
        data = extract_json(raw)
        if not isinstance(data, dict):
            return ir
        if isinstance(data.get("headline"), str) and data["headline"].strip() and not ir.header.headline:
            ir.header.headline = data["headline"].strip()[:160]
        for r in data.get("experience_repairs", []) or []:
            if not isinstance(r, dict):
                continue
            i = r.get("index")
            if not isinstance(i, int) or not (0 <= i < len(ir.experience)):
                continue
            blk = ir.experience[i]
            if r.get("title") and not blk.title:
                blk.title = str(r["title"])[:120]
            if r.get("company") and not blk.company:
                blk.company = str(r["company"])[:120]
            if r.get("dates") and not blk.dates:
                blk.dates = str(r["dates"])[:60]
        for r in data.get("education_repairs", []) or []:
            if not isinstance(r, dict):
                continue
            i = r.get("index")
            if not isinstance(i, int) or not (0 <= i < len(ir.education)):
                continue
            ed = ir.education[i]
            if r.get("institution") and not ed.institution:
                ed.institution = str(r["institution"])[:160]
            if r.get("location") and not ed.location:
                ed.location = str(r["location"])[:120]
        for ms in data.get("missed_sections", []) or []:
            if not isinstance(ms, dict):
                continue
            title = str(ms.get("title", "")).strip()
            items = ms.get("items") or []
            if title and isinstance(items, list):
                ir.extras.append(ExtraSection(
                    title=title,
                    items=[str(x).strip() for x in items if str(x).strip()],
                ))
        return ir
