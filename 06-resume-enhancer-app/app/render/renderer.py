"""
renderer.py - ResumeIR -> .tex string.

Uses a Jinja2 environment with non-LaTeX-conflicting block delimiters
((( ... ))) and ((* ... *)) so the template can sit alongside real
LaTeX commands without escaping headaches.

The `tex` filter escapes user content; `url` minimally sanitises hrefs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .. import __version__
from ..core.config import TEMPLATE_DIR
from ..core.ir import ResumeIR

log = logging.getLogger(__name__)


# Order matters - the backslash must be replaced first
_TEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


# Markdown-ish input from the LLM that we want to preserve as semantic
# emphasis rather than literal asterisks. Run BEFORE escaping so the
# backslashes we add survive.
_MD_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")
_MD_ITAL_RE = re.compile(r"(?<![a-zA-Z*])\*([^*\n]+?)\*(?![a-zA-Z*])")


def _filter_tex(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    # Convert markdown emphasis to LaTeX BEFORE escape, using sentinel
    # tokens that won't be touched by the literal escape pass below.
    s = _MD_BOLD_RE.sub(r"BOLDBEGIN\1BOLDEND", s)
    s = _MD_ITAL_RE.sub(r"ITALBEGIN\1ITALEND", s)
    for src, dst in _TEX_ESCAPES:
        s = s.replace(src, dst)
    s = s.replace("BOLDBEGIN", r"\textbf{").replace("BOLDEND", r"}")
    s = s.replace("ITALBEGIN", r"\textit{").replace("ITALEND", r"}")
    # Common typographic substitutions
    s = s.replace(" ", " ")
    return s


def _filter_url(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    # Allow only safe URL characters. Strip everything weird; if a `#`
    # or `_` appears in a URL it must remain literal (no escape).
    return re.sub(r"[^A-Za-z0-9:/._\-?&=%#@+]", "", s)


_env: Environment | None = None


def _get_env() -> Environment:
    global _env
    if _env is None:
        _env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            block_start_string="((*",
            block_end_string="*))",
            variable_start_string="(((",
            variable_end_string=")))",
            comment_start_string="((#",
            comment_end_string="#))",
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            undefined=StrictUndefined,
        )
        _env.filters["tex"] = _filter_tex
        _env.filters["url"] = _filter_url
    return _env


def render_ir_to_tex(ir: ResumeIR) -> str:
    env = _get_env()
    template = env.get_template("template.tex.j2")
    out = template.render(
        ir=ir,
        header=ir.header,
        version=__version__,
    )
    # Collapse runs of blank lines that the trim_blocks logic can leave behind
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def render_ir_to_file(ir: ResumeIR, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_ir_to_tex(ir), encoding="utf-8")
    return p


def validate_tex(tex: str) -> list[str]:
    """Light structural checks on generated LaTeX. Returns list of warnings."""
    issues: list[str] = []
    if r"\begin{document}" not in tex:
        issues.append("Missing \\begin{document}")
    if r"\end{document}" not in tex:
        issues.append("Missing \\end{document}")
    # Unmatched braces — count { vs }
    open_b = tex.count("{") - tex.count(r"\{")
    close_b = tex.count("}") - tex.count(r"\}")
    if abs(open_b - close_b) > 4:
        issues.append(f"Brace mismatch: {open_b} open vs {close_b} close — check for truncated output")
    # Sentinel bleed-through
    for sentinel in ("BOLDBEGIN", "BOLDEND", "ITALBEGIN", "ITALEND"):
        if sentinel in tex:
            issues.append(f"Rendering artefact '{sentinel}' leaked into output — bold/italic markup issue")
    # Placeholder bleed-through
    if "[YOUR FULL NAME]" in tex or "[ADD 2-3 LINE SUMMARY" in tex:
        issues.append("Unfilled placeholder text in output — name or summary is missing from your resume")
    return issues
