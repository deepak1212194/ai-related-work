"""
skills.py - markdown-driven skill loader.

Every agent's behaviour is encoded in a markdown file under skills/.
The loader caches parsed contents and exposes named blocks. Files are
hot-reloadable - call `reload_skills()` to pick up live edits without
restarting the service.

File format (markdown, with optional `## block_name` headings):

    # Title (free-form, ignored by the loader)

    Default block content goes here. Anything before the first `##`
    is the file's "default" block.

    ## block_name_one

    First named block.

    ## block_name_two

    Second named block.

The loader returns a dict {block_name: text}. When an agent asks for
a specific block, missing blocks fall through to the file's "default"
content rather than raising.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import SKILLS_DIR

log = logging.getLogger(__name__)


@dataclass
class SkillFile:
    name: str                          # filename without extension
    path: Path
    default: str = ""
    blocks: Dict[str, str] = field(default_factory=dict)
    loaded_at: float = 0.0


@dataclass
class SkillBundle:
    files: Dict[str, SkillFile] = field(default_factory=dict)
    role_files: Dict[str, SkillFile] = field(default_factory=dict)

    def get_block(self, file_name: str, block: str = "") -> str:
        f = self.files.get(file_name)
        if not f:
            return ""
        if block and block in f.blocks:
            return f.blocks[block]
        return f.default or "\n\n".join(f.blocks.values())

    def get_role(self, role_id: str) -> str:
        f = self.role_files.get(role_id)
        if not f:
            return ""
        return f.default or "\n\n".join(f.blocks.values())

    def get_role_blocks(self, role_id: str, block_names: List[str]) -> str:
        """Return only selected named blocks from a role profile.

        Used for prompt token control: include only high-signal sections
        (for example, priority keywords + hiring signals) instead of
        always injecting the full role markdown file.
        """
        f = self.role_files.get(role_id)
        if not f:
            return ""
        parts: List[str] = []
        for raw in block_names:
            key = raw.strip().lower().replace(" ", "_")
            body = f.blocks.get(key, "")
            if body:
                parts.append(f"## {key}\n{body}")
        return "\n\n".join(parts).strip()

    def list_roles(self) -> List[str]:
        return sorted(self.role_files.keys())

    def list_files(self) -> List[str]:
        return sorted(self.files.keys())


_BUNDLE: SkillBundle | None = None
_LOCK = threading.Lock()


def _parse_markdown(text: str) -> tuple[str, Dict[str, str]]:
    """Split a markdown file into (default, {block: text})."""
    blocks: Dict[str, str] = {}
    parts = re.split(r"(?m)^##\s+(.+?)\s*$", text)
    default = parts[0].strip()
    # parts: [default, name1, body1, name2, body2, ...]
    for i in range(1, len(parts), 2):
        name = parts[i].strip().lower().replace(" ", "_")
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        blocks[name] = body
    return default, blocks


def load_skills() -> SkillBundle:
    """Load all *.md files under skills/ into memory."""
    global _BUNDLE
    with _LOCK:
        bundle = SkillBundle()
        if not SKILLS_DIR.exists():
            log.warning("[skills] %s does not exist", SKILLS_DIR)
            _BUNDLE = bundle
            return bundle
        for path in sorted(SKILLS_DIR.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:                          # noqa: BLE001
                log.warning("[skills] could not read %s: %s", path, e)
                continue
            default, blocks = _parse_markdown(text)
            sf = SkillFile(name=path.stem, path=path, default=default, blocks=blocks)
            bundle.files[path.stem] = sf
            if path.stem.startswith("role_"):
                role_id = path.stem[len("role_"):]
                bundle.role_files[role_id] = sf
        _BUNDLE = bundle
        log.info(
            "[skills] loaded %d files, %d roles",
            len(bundle.files), len(bundle.role_files),
        )
        return bundle


def reload_skills() -> SkillBundle:
    return load_skills()


def get_bundle() -> SkillBundle:
    if _BUNDLE is None:
        return load_skills()
    return _BUNDLE
