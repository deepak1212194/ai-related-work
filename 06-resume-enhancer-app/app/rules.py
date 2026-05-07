"""
rules.py — Skill-File-Driven Enhancement Rules
=================================================

All rules are now loaded from `skills/*.md` files at runtime,
similar to how AI agents (Claude, etc.) use skill files for
context injection.

This module provides backward-compatible functions used by
enhancer.py and main.py, but all content comes from the
skill loader.

Architecture:
  skills/core_rules.md         → base system prompt (non-negotiables)
  skills/role_*.md             → per-role emphasis and keywords
  skills/section_tasks.md      → per-section task templates

To modify behavior: edit the .md files. No code changes needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .skill_loader import load_skills, RoleSkill

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Backward-compatible RoleProfile wrapper
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RoleProfile:
    id: str
    name: str
    description: str
    emphasis: str
    keywords: list[str] = field(default_factory=list)


def _skill_to_profile(skill: RoleSkill) -> RoleProfile:
    """Convert a loaded RoleSkill to a RoleProfile."""
    return RoleProfile(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        emphasis=skill.emphasis,
        keywords=skill.keywords,
    )


# ──────────────────────────────────────────────────────────────────────
# Public API — backward compatible
# ──────────────────────────────────────────────────────────────────────
def _get_profiles() -> dict[str, RoleProfile]:
    """Load role profiles from skill files."""
    skills = load_skills()
    profiles = {
        role_id: _skill_to_profile(role)
        for role_id, role in skills.roles.items()
    }
    # Ensure at least a default profile exists
    if not profiles:
        profiles["ai_ml_engineer"] = RoleProfile(
            id="ai_ml_engineer",
            name="AI / ML Engineer",
            description="LLMs, RAG, multi-agent, fine-tuning, computer vision",
            emphasis="TARGET ROLE: AI / ML Engineer",
            keywords=["LLM", "RAG", "PyTorch", "FAISS"],
        )
    return profiles


# Lazy-loaded profiles dict
ROLE_PROFILES: dict[str, RoleProfile] = {}


def _ensure_loaded():
    global ROLE_PROFILES
    if not ROLE_PROFILES:
        ROLE_PROFILES.update(_get_profiles())


def list_roles() -> list[dict]:
    """JSON-friendly listing for the UI to render the role picker."""
    _ensure_loaded()
    return [
        {"id": p.id, "name": p.name, "description": p.description}
        for p in ROLE_PROFILES.values()
    ]


def compose_system_prompt(role_id: str) -> str:
    """
    Build the full system prompt from skill files:
    core_rules.md content + role-specific emphasis.
    """
    _ensure_loaded()
    skills = load_skills()

    # Core rules from skill file
    core = skills.core_rules or ""

    # Role emphasis
    profile = ROLE_PROFILES.get(role_id)
    if profile is None:
        profile = ROLE_PROFILES.get("ai_ml_engineer")
    if profile is None:
        profile = next(iter(ROLE_PROFILES.values()), None)

    emphasis = profile.emphasis if profile else ""

    return f"{core}\n\n{emphasis}\n"


# ──────────────────────────────────────────────────────────────────────
# Section task templates — loaded from skill files
# ──────────────────────────────────────────────────────────────────────
def _get_task_template(task_name: str, content: str) -> str:
    """
    Get a section task template from skill files.
    Falls back to a simple default if not found.
    """
    skills = load_skills()
    template = skills.section_tasks.get(task_name, "")

    if template:
        return f"{template}\n\nInput:\n\"\"\"\n{content}\n\"\"\"\n\nOutput the enhanced text as plain text, no quotes, no preamble."

    # Fallback: simple generic template
    return (
        f"Task: enhance the following {task_name} section.\n"
        f"Preserve all facts, numbers, and keywords. Only strengthen.\n\n"
        f"Input:\n\"\"\"\n{content}\n\"\"\"\n\n"
        f"Output the enhanced text as plain text, no quotes, no preamble."
    )


# Backward-compatible task functions
def get_summary_task(content: str) -> str:
    return _get_task_template("summary", content)


def get_bullet_task(content: str) -> str:
    return _get_task_template("bullet", content)


def get_skills_task(content: str) -> str:
    return _get_task_template("skills", content)


def get_achievement_task(content: str) -> str:
    return _get_task_template("achievement", content)


# Legacy format-string templates for direct use
# These are populated lazily from skill files
SUMMARY_TASK = "{content}"  # Placeholder — actual template via get_summary_task()
BULLET_TASK = "{content}"
SKILLS_TASK = "{content}"
ACHIEVEMENT_TASK = "{content}"
