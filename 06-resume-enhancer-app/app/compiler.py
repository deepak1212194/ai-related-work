"""
compiler.py — LaTeX rendering and PDF compilation
==================================================

Two stages:

  1. render_tex(parsed)  — fills the canonical Jake-style template
                           in templates/base.tex with the enhanced
                           ParsedResume content. Returns LaTeX string.
  2. compile_pdf(tex_path) — runs pdflatex on the .tex file. Returns
                             the path to the resulting .pdf or None
                             if compilation failed (the .tex is still
                             usable).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import TEMPLATES_DIR, settings
from .schemas import ParsedResume

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Jinja env — uses LaTeX-friendly delimiters so we don't collide with
# normal {} in LaTeX
# ──────────────────────────────────────────────────────────────────────
_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    block_start_string="<%",  block_end_string="%>",
    variable_start_string="<<", variable_end_string=">>",
    comment_start_string="<#", comment_end_string="#>",
    trim_blocks=True, lstrip_blocks=True,
    autoescape=False,
)


# ──────────────────────────────────────────────────────────────────────
# LaTeX-safe escape — deliberately conservative
# ──────────────────────────────────────────────────────────────────────
_LATEX_REPLACEMENTS = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\textasciicircum{}"),
]


def latex_escape(s: str) -> str:
    if not s:
        return ""
    for a, b in _LATEX_REPLACEMENTS:
        s = s.replace(a, b)
    return s


_env.filters["tex"] = latex_escape


# ──────────────────────────────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────────────────────────────
def render_tex(parsed: ParsedResume) -> str:
    template = _env.get_template("base.tex")
    return template.render(r=parsed)


# ──────────────────────────────────────────────────────────────────────
# Compile
# ──────────────────────────────────────────────────────────────────────
def compile_pdf(tex_path: Path) -> Path | None:
    """
    Run pdflatex (or tectonic) on tex_path. Returns path to the .pdf,
    or None if the compiler is unavailable or compilation errored. The
    .tex file is always usable regardless.
    """
    cmd = settings.pdflatex_cmd
    if shutil.which(cmd) is None:
        log.warning("[COMPILE] %s not on PATH — skipping PDF compile", cmd)
        return None

    work_dir = tex_path.parent
    log.info("[COMPILE] %s on %s", cmd, tex_path.name)
    try:
        result = subprocess.run(
            [cmd, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=work_dir,
            capture_output=True,
            timeout=settings.compile_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        log.error("[COMPILE] timed out after %ds", settings.compile_timeout_seconds)
        return None

    pdf_path = tex_path.with_suffix(".pdf")
    if result.returncode != 0 or not pdf_path.exists():
        log.error("[COMPILE] failed (rc=%d) — first 400 chars of stdout follow",
                  result.returncode)
        log.error(result.stdout.decode("utf-8", errors="replace")[:400])
        return None

    return pdf_path
