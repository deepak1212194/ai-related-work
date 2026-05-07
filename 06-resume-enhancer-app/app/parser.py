"""
parser.py — PDF + .tex resume parser
======================================

Two entry points:

  parse_pdf(path)  — reads a PDF, extracts text, runs section-aware
                     heuristics to populate a ParsedResume.
  parse_tex(path)  — reads a .tex file, looks for the standard section
                     markers (Summary / Skills / Experience / Education
                     / Achievements) and extracts content.

Heuristics, not AST parsing — by design. Resumes have wildly varied
LaTeX templates; we read the section TEXT and rebuild from the canonical
template at output time. This is the deliberate "restrictive" choice —
we never try to mutate the input .tex in place.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from .schemas import EducationBlock, ExperienceBlock, ParsedResume

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Section marker patterns (case-insensitive)
# ──────────────────────────────────────────────────────────────────────
SECTION_HEADERS = [
    "professional summary", "summary", "about",
    "technical skills", "skills",
    "professional experience", "work experience", "experience",
    "education",
    "achievements", "achievements & recognition", "honors", "certifications",
    "patent", "publications",
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ──────────────────────────────────────────────────────────────────────
# PDF parsing
# ──────────────────────────────────────────────────────────────────────
def parse_pdf(path: Path) -> ParsedResume:
    """Extract section text from a PDF resume using a header-block split."""
    log.info("[PARSE] PDF: %s", path)
    doc = fitz.open(path)
    raw_lines: list[str] = []
    for page in doc:
        for line in page.get_text("text").splitlines():
            line = line.rstrip()
            if line.strip():
                raw_lines.append(line)
    doc.close()

    if not raw_lines:
        raise ValueError("PDF appears to be empty or unreadable")

    name = _normalize(raw_lines[0])
    contact_line = ""
    body_start = 1
    # Heuristic: first 1-3 lines after name are contact info
    for i in range(1, min(5, len(raw_lines))):
        l = raw_lines[i].lower()
        if any(s in l for s in ("@", "linkedin", "github", "+", "phone")):
            contact_line += " " + raw_lines[i]
            body_start = i + 1
        else:
            break
    contact_line = _normalize(contact_line)

    body = raw_lines[body_start:]
    sections = _split_into_sections(body)

    parsed = ParsedResume(name=name, contact_line=contact_line)
    parsed.summary = sections.get("summary", "") or sections.get("professional summary", "")
    parsed.skills = _parse_skills(
        sections.get("technical skills") or sections.get("skills") or ""
    )
    parsed.experience_blocks = _parse_experience(
        sections.get("professional experience")
        or sections.get("work experience")
        or sections.get("experience")
        or ""
    )
    parsed.education_blocks = _parse_education(sections.get("education", ""))
    parsed.achievements = _parse_achievements(
        sections.get("achievements & recognition")
        or sections.get("achievements")
        or sections.get("honors")
        or sections.get("certifications")
        or ""
    )
    return parsed


def _split_into_sections(lines: list[str]) -> dict[str, str]:
    """Group lines into a {section_header_lowercased: body} dict."""
    out: dict[str, list[str]] = {}
    current = "header"
    out[current] = []
    for line in lines:
        norm = line.strip().lower()
        if norm in SECTION_HEADERS:
            current = norm
            out[current] = []
        else:
            out.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def _parse_skills(text: str) -> dict[str, str]:
    """Skills lines often look like 'Bucket: items, items, items'."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            bucket, items = line.split(":", 1)
            out[_normalize(bucket)] = _normalize(items)
    return out


def _parse_experience(text: str) -> list[ExperienceBlock]:
    """
    Heuristic: a line starting with a non-bullet character that contains
    a date range starts a new block; following bullets attach to it.
    """
    blocks: list[ExperienceBlock] = []
    current: ExperienceBlock | None = None
    date_re = re.compile(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
        r"\d{4}\s*[-–—]\s*(present|\d{4}))",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        is_bullet = line.lstrip().startswith(("•", "-", "▪", "▸", "*"))
        if not is_bullet and date_re.search(line):
            # New block
            if current:
                blocks.append(current)
            parts = re.split(r"\s{2,}|\t", line)
            current = ExperienceBlock(
                title=parts[0] if parts else line,
                company=parts[1] if len(parts) > 1 else "",
                dates=parts[-1] if len(parts) > 2 else "",
            )
        elif current is not None:
            cleaned = line.lstrip("•-▪▸* ").strip()
            if cleaned:
                current.bullets.append(cleaned)
    if current:
        blocks.append(current)
    return blocks


def _parse_education(text: str) -> list[EducationBlock]:
    blocks: list[EducationBlock] = []
    for line in text.splitlines():
        if "M.Tech" in line or "B.Tech" in line or "Bachelor" in line or "Master" in line or "PhD" in line:
            parts = re.split(r"\s{2,}|\|", line)
            blocks.append(EducationBlock(
                degree=parts[0].strip(),
                institution=parts[1].strip() if len(parts) > 1 else "",
                dates=parts[-1].strip() if len(parts) > 2 else "",
            ))
    return blocks


def _parse_achievements(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        cleaned = line.lstrip("•-▪▸* ").strip()
        if cleaned:
            out.append(cleaned)
    return out


# ──────────────────────────────────────────────────────────────────────
# .tex parsing — section-aware regex extraction
# ──────────────────────────────────────────────────────────────────────
SECTION_RE = re.compile(r"\\section\{([^}]+)\}")
SECTION_BODY_RE = re.compile(
    r"\\section\{(?P<title>[^}]+)\}(?P<body>.*?)(?=\\section\{|\\end\{document\})",
    re.DOTALL,
)


def parse_tex(path: Path) -> ParsedResume:
    """
    Extract section bodies from a .tex resume. Returns the same
    ParsedResume shape so the downstream enhancer is parser-agnostic.
    """
    log.info("[PARSE] .tex: %s", path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Try to grab the candidate's name from the \Huge \textbf{...} pattern
    name_match = re.search(r"\\Huge[^{]*\\textbf\{[^}]*?([\w][^}\\]+)\}", text)
    name = name_match.group(1).strip() if name_match else ""

    parsed = ParsedResume(name=name)
    sections: dict[str, str] = {}
    for m in SECTION_BODY_RE.finditer(text):
        sections[m.group("title").strip().lower()] = m.group("body").strip()

    parsed.summary = _strip_latex(
        sections.get("professional summary") or sections.get("summary") or ""
    )
    parsed.skills = _parse_tex_skills(
        sections.get("technical skills") or sections.get("skills") or ""
    )
    parsed.experience_blocks = _parse_tex_experience(
        sections.get("professional experience")
        or sections.get("experience")
        or ""
    )
    parsed.education_blocks = _parse_tex_education(sections.get("education", ""))
    parsed.achievements = _parse_tex_achievements(
        sections.get("achievements & recognition")
        or sections.get("achievements")
        or ""
    )
    return parsed


def _strip_latex(s: str) -> str:
    """Lossy cleanup — drop LaTeX commands so the LLM sees plain text."""
    s = re.sub(r"\\href\{[^}]+\}\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\\\(\s|$)", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"[\{\}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_tex_skills(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    # Pattern: \textbf{Bucket}{: items, items}
    for m in re.finditer(
        r"\\textbf\{([^}]+)\}\{:\s*([^}]+)\}", body
    ):
        out[m.group(1).strip()] = _strip_latex(m.group(2))
    return out


def _parse_tex_experience(body: str) -> list[ExperienceBlock]:
    blocks: list[ExperienceBlock] = []
    # \resumeSubheading{title}{dates}{company}{location}
    sub_re = re.compile(
        r"\\resumeSubheading\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}"
    )
    item_re = re.compile(r"\\resumeItem\{(.+?)\}", re.DOTALL)

    chunks = re.split(r"\\resumeSubheading", body)
    for chunk in chunks[1:]:
        # Re-prefix so sub_re matches
        m = sub_re.search("\\resumeSubheading" + chunk)
        if not m:
            continue
        title, dates, company, location = (s.strip() for s in m.groups())
        bullets = [
            _strip_latex(b.group(1)) for b in item_re.finditer(chunk)
        ]
        blocks.append(ExperienceBlock(
            title=title, dates=dates, company=company,
            location=location, bullets=bullets,
        ))
    return blocks


def _parse_tex_education(body: str) -> list[EducationBlock]:
    blocks: list[EducationBlock] = []
    # Try to pick out \textbf{...} ... textit{years}
    for m in re.finditer(
        r"\\textbf\{([^}]+)\}.*?\\textit\{([^}]+)\}.*?\\small\s*([^}]+)\}",
        body, re.DOTALL,
    ):
        degree, dates, institution = m.groups()
        blocks.append(EducationBlock(
            degree=_strip_latex(degree),
            dates=_strip_latex(dates),
            institution=_strip_latex(institution),
        ))
    return blocks


def _parse_tex_achievements(body: str) -> list[str]:
    out = []
    for m in re.finditer(
        r"\\achieveRow\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}", body
    ):
        title, desc, year = (_strip_latex(s) for s in m.groups())
        out.append(f"{title} | {desc} | {year}")
    return out
