"""
skill_loader.py — Skill File Reader
======================================
Resume Enhancer — Skill System

Reads `.md` skill files from the `skills/` directory at runtime,
similar to how Claude uses skill files for agent instructions.

The skill loader:
  1. Reads core_rules.md → base system prompt
  2. Reads role_*.md → role-specific emphasis
  3. Reads section_tasks.md → per-section task templates
  4. Caches loaded skills in memory for fast access
  5. Supports hot-reload: call reload() to pick up edits

This architecture means you can change how the agent enhances
resumes by editing markdown files — no code changes needed.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RoleSkill:
    """A role profile loaded from a skill file."""
    id: str
    name: str
    description: str
    emphasis: str
    keywords: List[str] = field(default_factory=list)
    section_guidance: Dict[str, str] = field(default_factory=dict)


@dataclass
class SectionTask:
    """A per-section task template."""
    section: str
    template: str


@dataclass
class SkillSet:
    """All loaded skills — the agent's full instruction set."""
    core_rules: str = ""
    roles: Dict[str, RoleSkill] = field(default_factory=dict)
    section_tasks: Dict[str, str] = field(default_factory=dict)
    loaded_files: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Markdown section parser
# ──────────────────────────────────────────────────────────────────────
def _parse_md_sections(text: str) -> Dict[str, str]:
    """
    Parse a markdown file into sections keyed by heading.
    Returns {heading_text: content_below_heading}.
    """
    sections: Dict[str, str] = {}
    current_heading = ""
    current_lines: List[str] = []

    for line in text.split("\n"):
        heading_match = re.match(r'^#{1,3}\s+(.+)$', line)
        if heading_match:
            # Save previous section
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_heading:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _extract_list_items(text: str) -> List[str]:
    """Extract bullet/dash list items from markdown text."""
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            items.append(line[2:].strip())
        elif re.match(r'^\d+\.\s', line):
            items.append(re.sub(r'^\d+\.\s*', '', line).strip())
    return items


def _extract_keywords(text: str) -> List[str]:
    """Extract comma-separated keywords from a text block."""
    # Remove markdown formatting
    clean = re.sub(r'[*_`]', '', text)
    # Split by comma or newline
    items = re.split(r'[,\n]', clean)
    return [k.strip() for k in items if k.strip() and len(k.strip()) > 1]


# ──────────────────────────────────────────────────────────────────────
# Skill file loaders
# ──────────────────────────────────────────────────────────────────────
def _load_core_rules(skills_dir: Path) -> str:
    """Load core_rules.md as the base system prompt."""
    path = skills_dir / "core_rules.md"
    if not path.exists():
        log.warning("[SKILL] core_rules.md not found — using empty rules")
        return ""

    text = path.read_text(encoding="utf-8")

    # Extract the actual rules content (skip the title and blockquote)
    sections = _parse_md_sections(text)

    parts = []
    # Use Identity section
    if "Identity" in sections:
        parts.append(sections["Identity"])

    # Use Enhancement Rules
    for key in sections:
        if "enhancement" in key.lower() or "rules" in key.lower():
            parts.append(sections[key])

    # Use Safety Constraints
    if "Safety Constraints" in sections:
        parts.append(sections["Safety Constraints"])

    # Use ATS Optimization
    if "ATS Optimization" in sections:
        parts.append(sections["ATS Optimization"])

    if not parts:
        # Fallback: use the whole file minus the title
        lines = text.split("\n")
        parts = ["\n".join(l for l in lines if not l.startswith("#") and not l.startswith(">"))]

    log.info("[SKILL] Loaded core_rules.md (%d chars)", sum(len(p) for p in parts))
    return "\n\n".join(parts)


def _load_role_skill(path: Path) -> Optional[RoleSkill]:
    """Load a role_*.md file into a RoleSkill."""
    try:
        text = path.read_text(encoding="utf-8")
        sections = _parse_md_sections(text)

        # Extract role ID from filename: role_ai_ml_engineer.md → ai_ml_engineer
        role_id = path.stem.replace("role_", "")

        # Parse target role
        target = sections.get("Target Role", "").strip()
        name = target.split("(")[0].strip() if target else role_id.replace("_", " ").title()
        description = ""

        # Parse emphasis
        emphasis_parts = []
        emphasis_text = sections.get("What to Emphasize", "")
        if emphasis_text:
            emphasis_parts.append(f"TARGET ROLE: {target}\n" if target else "")
            emphasis_parts.append(emphasis_text)

        # Parse keywords
        keywords = []
        kw_text = sections.get("Priority Keywords", "")
        if kw_text:
            keywords = _extract_keywords(kw_text)

        # Parse section-specific guidance
        guidance = {}
        for key, content in sections.items():
            if key in ("Summary", "Experience Bullets", "Skills", "Education", "Achievements"):
                guidance[key] = content

        # Build description from first line of emphasis
        if emphasis_text:
            first_items = _extract_list_items(emphasis_text)[:2]
            description = "; ".join(first_items) if first_items else emphasis_text[:80]

        return RoleSkill(
            id=role_id,
            name=name,
            description=description,
            emphasis="\n".join(emphasis_parts),
            keywords=keywords,
            section_guidance=guidance,
        )
    except Exception as e:
        log.warning("[SKILL] Failed to load %s: %s", path.name, e)
        return None


def _load_section_tasks(skills_dir: Path) -> Dict[str, str]:
    """Load section_tasks.md into task templates."""
    path = skills_dir / "section_tasks.md"
    if not path.exists():
        log.warning("[SKILL] section_tasks.md not found — using defaults")
        return {}

    text = path.read_text(encoding="utf-8")
    sections = _parse_md_sections(text)

    tasks = {}
    for key, content in sections.items():
        # Normalize key: "Summary Task" → "summary"
        task_name = key.lower().replace(" task", "").strip()
        if content.strip():
            tasks[task_name] = content.strip()

    log.info("[SKILL] Loaded %d section tasks", len(tasks))
    return tasks


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
_cached_skills: Optional[SkillSet] = None


def load_skills(skills_dir: Optional[Path] = None) -> SkillSet:
    """
    Load all skill files from the skills directory.
    Results are cached; call reload() to refresh.
    """
    global _cached_skills
    if _cached_skills is not None:
        return _cached_skills

    sd = skills_dir or SKILLS_DIR
    sd.mkdir(parents=True, exist_ok=True)

    skills = SkillSet()

    # Core rules
    skills.core_rules = _load_core_rules(sd)

    # Role skills
    for role_file in sorted(sd.glob("role_*.md")):
        role = _load_role_skill(role_file)
        if role:
            skills.roles[role.id] = role
            skills.loaded_files.append(role_file.name)

    # Section tasks
    skills.section_tasks = _load_section_tasks(sd)

    if (sd / "core_rules.md").exists():
        skills.loaded_files.insert(0, "core_rules.md")
    if (sd / "section_tasks.md").exists():
        skills.loaded_files.append("section_tasks.md")

    log.info(
        "[SKILL] Loaded %d files: %d roles, %d tasks",
        len(skills.loaded_files), len(skills.roles), len(skills.section_tasks),
    )

    _cached_skills = skills
    return skills


def reload_skills() -> SkillSet:
    """Force-reload all skill files (useful after editing .md files)."""
    global _cached_skills
    _cached_skills = None
    return load_skills()


def get_skill_info() -> Dict:
    """Return skill loading status for health/debug endpoints."""
    skills = load_skills()
    return {
        "loaded_files": skills.loaded_files,
        "roles": list(skills.roles.keys()),
        "section_tasks": list(skills.section_tasks.keys()),
        "core_rules_length": len(skills.core_rules),
    }
