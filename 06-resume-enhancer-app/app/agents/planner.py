"""
PlannerAgent - decides section order, skill ordering, and lead-bullet hints.

Uses the LLM with strict JSON output and falls back to a deterministic
default plan if the LLM call fails or returns invalid data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from ..core.ir import ResumeIR
from .base import Agent, extract_json

log = logging.getLogger(__name__)


DEFAULT_ORDER = [
    "summary", "skills", "experience", "projects",
    "education", "certifications", "achievements",
    "publications", "extras",
]


# Per-role default ordering. Used when the LLM is unavailable.
ROLE_DEFAULT_ORDERS: Dict[str, List[str]] = {
    "ai_ml_engineer": [
        "summary", "skills", "experience", "projects",
        "achievements", "publications", "education",
        "certifications", "extras",
    ],
    "data_scientist": [
        "summary", "skills", "experience", "projects",
        "publications", "education", "achievements",
        "certifications", "extras",
    ],
    "software_engineer": [
        "summary", "skills", "experience", "projects",
        "education", "certifications", "achievements",
        "publications", "extras",
    ],
    "devops_cloud_engineer": [
        "summary", "skills", "experience", "certifications",
        "projects", "education", "achievements",
        "publications", "extras",
    ],
    "product_manager": [
        "summary", "experience", "projects", "skills",
        "education", "achievements", "certifications",
        "publications", "extras",
    ],
}


@dataclass
class Plan:
    section_order: List[str] = field(default_factory=lambda: list(DEFAULT_ORDER))
    skill_order_indices: List[int] = field(default_factory=list)
    experience_order_indices: List[int] = field(default_factory=list)
    lead_bullet_hints: Dict[int, List[int]] = field(default_factory=dict)


def _default_plan(ir: ResumeIR, role_id: str) -> Plan:
    return Plan(
        section_order=ROLE_DEFAULT_ORDERS.get(role_id, list(DEFAULT_ORDER)),
        skill_order_indices=list(range(len(ir.skills))),
        experience_order_indices=list(range(len(ir.experience))),
        lead_bullet_hints={
            i: [0] if blk.bullets else []
            for i, blk in enumerate(ir.experience)
        },
    )


class PlannerAgent(Agent):
    def plan(self, ir: ResumeIR, role_id: str) -> Plan:
        plan = _default_plan(ir, role_id)
        try:
            sys_prompt = (
                self.skills.get_block("00_core_rules")
                + "\n\n----\n\n"
                + self.skills.get_block("03_planning")
                + "\n\n----\n\n"
                + self.skills.get_role(role_id)
            )
            n_skills = len(ir.skills)
            n_exp = len(ir.experience)
            user = (
                f"TARGET_ROLE: {role_id}\n\n"
                f"AVAILABLE_SECTIONS: {[k for k in DEFAULT_ORDER if getattr(ir, k, None) or (k=='extras' and ir.extras)]}\n"
                f"NUM_SKILL_BUCKETS: {n_skills}\n"
                f"SKILL_BUCKET_NAMES: {[s.name for s in ir.skills]}\n"
                f"NUM_EXPERIENCE_BLOCKS: {n_exp}\n"
                f"EXPERIENCE_TITLES: {[(e.title, e.company, e.dates) for e in ir.experience]}\n\n"
                "Return JSON only."
            )
            raw = self.llm.complete(sys_prompt, user, max_tokens=600, temperature=0.1)
        except Exception as e:                              # noqa: BLE001
            log.warning("[Planner] falling back to default plan: %s", e)
            return plan
        data = extract_json(raw)
        if not isinstance(data, dict):
            return plan
        # section_order
        order = data.get("section_order")
        if isinstance(order, list):
            cleaned = [s for s in order if isinstance(s, str) and s in DEFAULT_ORDER]
            # Append any defaults that the LLM forgot
            for s in DEFAULT_ORDER:
                if s not in cleaned:
                    cleaned.append(s)
            plan.section_order = cleaned
        # skill ordering
        soi = data.get("skill_order_indices")
        if isinstance(soi, list) and ir.skills:
            valid = [i for i in soi if isinstance(i, int) and 0 <= i < len(ir.skills)]
            for i in range(len(ir.skills)):
                if i not in valid:
                    valid.append(i)
            plan.skill_order_indices = valid
        # experience ordering
        eoi = data.get("experience_order_indices")
        if isinstance(eoi, list) and ir.experience:
            valid = [i for i in eoi if isinstance(i, int) and 0 <= i < len(ir.experience)]
            for i in range(len(ir.experience)):
                if i not in valid:
                    valid.append(i)
            plan.experience_order_indices = valid
        # lead bullet hints
        lbh = data.get("lead_bullet_hints")
        if isinstance(lbh, dict):
            plan.lead_bullet_hints = {}
            for k, v in lbh.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                if not (0 <= idx < len(ir.experience)):
                    continue
                if isinstance(v, list):
                    plan.lead_bullet_hints[idx] = [
                        i for i in v
                        if isinstance(i, int) and 0 <= i < len(ir.experience[idx].bullets)
                    ][:2]
        return plan


def apply_plan(ir: ResumeIR, plan: Plan) -> ResumeIR:
    """Apply a plan to an IR (mutates the IR in place and returns it)."""
    if plan.section_order:
        ir.section_order = list(plan.section_order)
    if plan.skill_order_indices and len(plan.skill_order_indices) == len(ir.skills):
        ir.skills = [ir.skills[i] for i in plan.skill_order_indices]
    if plan.experience_order_indices and len(plan.experience_order_indices) == len(ir.experience):
        ir.experience = [ir.experience[i] for i in plan.experience_order_indices]
    return ir
