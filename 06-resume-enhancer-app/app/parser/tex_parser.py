"""
tex_parser.py - .tex resume -> ResumeIR.

Heuristic, regex-driven, intentionally NOT a full LaTeX parser. Real
resumes use a small set of common templates (Jake's Resume, Awesome-CV,
ModernCV, custom Overleaf forks); this parser handles the macro
conventions of all of them via a uniform section-and-token walk.

Strategy:
  1. Strip comments + preamble noise.
  2. Pull header (name, headline, links, location) from the first
     center / heading block.
  3. Walk \\section{Title}...\\section{Title} / \\end{document} to get
     section bodies.
  4. Per-section structured extractors with PLAIN-TEXT fallbacks - so a
     user-written \\subsection or naked itemize still survives.

Everything that doesn't match a known section becomes an `ExtraSection`
so no input content is lost, even if we can't classify it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.ir import (
    AchievementItem, CertificationItem, ContactLink, EducationBlock,
    ExperienceBlock, ExperienceGroup, ExtraSection, HeaderInfo,
    ProjectBlock, PublicationItem, ResumeIR, SkillBucket,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Section-name normalisation
# ----------------------------------------------------------------------
_SECTION_ALIASES: Dict[str, str] = {
    "summary": "summary",
    "professional summary": "summary",
    "career summary": "summary",
    "objective": "summary",
    "profile": "summary",
    "about": "summary",
    "about me": "summary",
    "skills": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "key skills": "skills",
    "expertise": "skills",
    "experience": "experience",
    "professional experience": "experience",
    "work experience": "experience",
    "employment": "experience",
    "employment history": "experience",
    "career history": "experience",
    "projects": "projects",
    "selected projects": "projects",
    "side projects": "projects",
    "personal projects": "projects",
    "open source": "projects",
    "academic projects": "projects",
    "education": "education",
    "academic background": "education",
    "qualifications": "education",
    "certifications": "certifications",
    "licenses & certifications": "certifications",
    "courses & certifications": "certifications",
    "courses and certifications": "certifications",
    "credentials": "certifications",
    "achievements": "achievements",
    "achievements & recognition": "achievements",
    "honors": "achievements",
    "awards": "achievements",
    "honors & awards": "achievements",
    "publications": "publications",
    "papers": "publications",
    "patents": "publications",
}


def _normalize_section(title: str) -> str:
    return _SECTION_ALIASES.get(title.strip().lower(), "")


# ----------------------------------------------------------------------
# LaTeX text cleanup
# ----------------------------------------------------------------------
_KEEP_CONTENT_WRAPPERS = [
    "textbf", "textit", "emph", "textsf", "texttt", "textsc",
    "small", "large", "Large", "LARGE", "Huge", "huge",
    "footnotesize", "scriptsize", "tiny", "normalsize",
    "itshape", "bfseries", "mdseries", "rmfamily", "sffamily", "ttfamily",
    "underline", "cvlistitem", "cvitem", "cvcolumncell", "cventry",
]
_KEEP_CONTENT_RE = re.compile(
    r"\\(" + "|".join(_KEEP_CONTENT_WRAPPERS) + r")\*?\{([^{}]*)\}"
)


def _strip_tex_comments(text: str) -> str:
    out: List[str] = []
    for line in text.splitlines():
        cleaned: List[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                cleaned.append(ch); cleaned.append(line[i + 1]); i += 2
                continue
            if ch == "%":
                break
            cleaned.append(ch); i += 1
        out.append("".join(cleaned).rstrip())
    return "\n".join(out)


def _strip_latex(s: str) -> str:
    if not s:
        return ""
    # Math-mode bullets
    s = re.sub(r"\$\\bullet\$", "*", s)
    s = re.sub(r"\\textbullet\b", "*", s)
    s = re.sub(r"\\quad\b", " ", s)
    s = re.sub(r"\\enspace\b", " ", s)
    s = re.sub(r"\\dots\b", "...", s)
    s = re.sub(r"\\ldots\b", "...", s)
    s = re.sub(r"\\[ ,;:!]", " ", s)
    s = s.replace(r"\&", "&").replace(r"\%", "%").replace(r"\#", "#")
    s = s.replace(r"\_", "_").replace(r"\$", "$").replace(r"\textasciitilde", "~")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\href\{[^}]+\}\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\url\{([^}]+)\}", r"\1", s)
    # Numeric superscripts -> Unicode
    sup = {"0": "⁰", "1": "¹", "2": "²", "3": "³",
           "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
           "8": "⁸", "9": "⁹"}
    s = re.sub(
        r"\\textsuperscript\{([^}]+)\}",
        lambda m: "".join(sup.get(c, c) for c in m.group(1)) if all(c in sup for c in m.group(1)) else m.group(1),
        s,
    )
    # Repeatedly unwrap nested formatting commands
    prev = None
    while prev != s:
        prev = s
        s = _KEEP_CONTENT_RE.sub(r"\2", s)
    s = re.sub(r"\\\\(\s|$)", r"\1", s)
    s = re.sub(r"\\color\{[^}]+\}", "", s)
    # Drop remaining \cmd[...]{...} and standalone \cmd
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"[\{\}]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n", "\n", s).strip()
    return s


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------
# Header extraction
# ----------------------------------------------------------------------
_NAME_PATTERNS = [
    # \documentTitle{Full Name}{...} — some custom Overleaf templates
    re.compile(r"\\documentTitle\s*\{([^}]+)\}"),
    re.compile(r"\\Huge\s*\\textbf\{\s*\\color\{[^}]+\}\s*([^}]+?)\s*\}"),
    re.compile(r"\\Huge\s*\\textbf\{\s*([^}\\]+?)\s*\}"),
    re.compile(r"\\Huge\s+([^\\\n}]+?)(?=\s*\\\\)"),
    re.compile(r"\{\\Huge\s+\\bf\s+([^}\\]+?)\}"),
    re.compile(r"\\name\s*\{([^}]+)\}"),
]


def _extract_name(text: str) -> str:
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = _strip_latex(m.group(1))
            if cand and len(cand) < 120:
                return cand
    return ""


def _extract_headline(text: str) -> str:
    """The small grey subtitle line under the name."""
    pats = [
        re.compile(
            r"\\Huge[^\n]*?\\\\\s*\\vspace\{[^}]*\}\s*\{?\\small[^}]*?\\color\{[^}]+\}\s*([^}]+?)\}",
            re.DOTALL,
        ),
        re.compile(
            r"\\Huge[^\n]*?\\\\\s*\\vspace\{[^}]*\}\s*\{?\\small\s*([^}\\]+?)\}",
            re.DOTALL,
        ),
    ]
    for pat in pats:
        m = pat.search(text)
        if m:
            return _strip_latex(m.group(1))
    return ""


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().\-]{7,}\d")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[\w\-_/.]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-_/.]+", re.I)
_TWITTER_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:twitter|x)\.com/[\w\-_/.]+", re.I)
_SCHOLAR_RE = re.compile(r"(?:https?://)?scholar\.google\.com/[\w\-_/.?=&%+]+", re.I)


def _extract_links(text: str) -> Tuple[List[ContactLink], str]:
    links: List[ContactLink] = []
    seen: set[str] = set()

    def _add(kind: str, label: str, url: str, icon: str) -> None:
        if kind in ("email", "phone", "linkedin", "github", "scholar", "twitter", "location"):
            key = kind
        else:
            key = f"{kind}:{(url or label).lower()}"
        if key in seen:
            return
        seen.add(key)
        links.append(ContactLink(kind=kind, label=label, url=url, icon=icon))

    # Pull \href explicitly first - they're the most reliable signal
    for m in re.finditer(r"\\href\{([^}]+)\}\{([^}]+)\}", text):
        url = m.group(1).strip()
        label = _strip_latex(m.group(2))
        label = re.sub(r"\\fa[A-Za-z]+\\?\s*", "", label).strip().lstrip("\\ ").strip()
        u = url.lower()
        if u.startswith("mailto:"):
            kind, icon = "email", "faEnvelope"; label = label or url[7:]
        elif "linkedin.com" in u:
            kind, icon = "linkedin", "faLinkedin"
        elif "github.com" in u:
            kind, icon = "github", "faGithub"
        elif "twitter.com" in u or "x.com" in u:
            kind, icon = "twitter", "faTwitter"
        elif "scholar.google.com" in u:
            kind, icon = "scholar", "faGraduationCap"
        else:
            kind, icon = "website", "faGlobe"
        _add(kind, label or url, url, icon)

    # Phone
    phone_m = re.search(r"\\faPhone\\?\s+([+0-9][\d\s().\-]{6,}\d)", text)
    if phone_m:
        ph = phone_m.group(1).strip()
        _add("phone", ph, "tel:" + re.sub(r"[^+0-9]", "", ph), "faPhone")
    else:
        # Fall back to a generic phone scan in the first 60 lines
        head = "\n".join(text.splitlines()[:60])
        m = _PHONE_RE.search(head)
        if m:
            digits = re.sub(r"[^0-9]", "", m.group(0))
            if len(digits) >= 9:
                _add("phone", m.group(0).strip(), "tel:" + digits, "faPhone")

    # Location via \faMapMarker
    loc_m = re.search(r"\\faMapMarker\\?\s+([^\\\n,]+)", text)
    location = ""
    if loc_m:
        location = _strip_latex(loc_m.group(1)).strip()
        _add("location", location, "", "faMapMarker")

    # Bare URLs that didn't appear in \href
    for pat, kind, icon in (
        (_LINKEDIN_RE, "linkedin", "faLinkedin"),
        (_GITHUB_RE, "github", "faGithub"),
        (_TWITTER_RE, "twitter", "faTwitter"),
        (_SCHOLAR_RE, "scholar", "faGraduationCap"),
    ):
        for m in pat.finditer(text):
            raw = m.group(0).rstrip(".,;|)")
            url = raw if raw.lower().startswith("http") else "https://" + raw
            _add(kind, raw, url, icon)

    # Email fallback (when not in \href{mailto:...})
    for m in _EMAIL_RE.finditer(text):
        addr = m.group(0)
        _add("email", addr, "mailto:" + addr, "faEnvelope")

    return links, location


# ----------------------------------------------------------------------
# Section splitting
# ----------------------------------------------------------------------
_SECTION_BODY_RE = re.compile(
    r"(?:\\section\*?|\\cvsection|\\rSection|\\tinysection)\{(?P<title>[^}]+)\}(?P<body>.*?)(?=(?:\\section\*?|\\cvsection|\\rSection|\\tinysection)\{|\\end\{document\})",
    re.DOTALL,
)


def _split_sections(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in _SECTION_BODY_RE.finditer(text):
        title = _strip_latex(m.group("title").strip())
        body = m.group("body").strip()
        out.append((title, body))
    # Fallback for templates that use \section* Title (without braces)
    if not out:
        loose = re.compile(
            r"\\(?:section|tinysection)\*?\s+(?P<title>[^\n\\]+)\n(?P<body>.*?)(?=\\(?:section|tinysection)\*?\s+[^\n\\]+|\\end\{document\})",
            re.DOTALL,
        )
        for m in loose.finditer(text):
            title = _strip_latex(m.group("title").strip())
            body = m.group("body").strip()
            out.append((title, body))
    return out


# ----------------------------------------------------------------------
# Skills
# ----------------------------------------------------------------------
def _parse_skills(body: str) -> List[SkillBucket]:
    out: List[SkillBucket] = []
    seen: set[str] = set()
    # Pattern: \textbf{Bucket}{: items}
    for m in re.finditer(r"\\textbf\s*\{([^}]+)\}\s*\{:\s*([^}]+)\}", body):
        name = _strip_latex(m.group(1)).strip().rstrip(":")
        items = [s.strip() for s in re.split(r",\s*(?![^()]*\))", _strip_latex(m.group(2))) if s.strip()]
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(SkillBucket(name=name, items=items))
    if out:
        return out
    # Fallback: line-by-line "Bucket: items"
    plain = _strip_latex(body)
    for line in plain.splitlines():
        if ":" in line:
            head, tail = line.split(":", 1)
            head = head.strip().lstrip("*-")
            items = [s.strip() for s in tail.split(",") if s.strip()]
            if head and head.lower() not in seen:
                seen.add(head.lower())
                out.append(SkillBucket(name=head, items=items))
    return out


# -- Brace-balanced argument extractor (replaces fragile regex) --
def _extract_brace_arg(text: str, start: int) -> Optional[Tuple[str, int]]:
    """Given that text[start] == '{', return (content, end_pos).

    Correctly handles arbitrarily nested braces so content like
    \\textbf{Azure OpenAI} inside a \\resumeItem{...} is captured fully.
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '\\':
            i += 2  # skip escaped char
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return None  # unbalanced


def _find_all_commands(text: str, cmd: str) -> List[Tuple[str, int, int]]:
    """Find all \\cmd{...} in text using brace-balanced extraction.

    Returns list of (content, start_of_cmd, end_of_closing_brace).
    Handles optional arguments like \\cmd[opt]{...}
    """
    results: List[Tuple[str, int, int]] = []
    pattern = re.compile(r'\\' + re.escape(cmd) + r'\*?(?:\s*\[[^\]]*\])?\s*\{')
    for m in pattern.finditer(text):
        brace_start = m.end() - 1  # position of the '{'
        extracted = _extract_brace_arg(text, brace_start)
        if extracted:
            content, end_pos = extracted
            results.append((content, m.start(), end_pos))
    return results


def _tokenize_experience_body(body: str) -> List[Tuple[str, str]]:
    """Walk an experience body and yield (kind, content) tokens.

    kind is 'group' or 'item'.
    Uses brace-balanced extraction so nested braces are handled.
    """
    tokens: List[Tuple[str, int, str]] = []  # (kind, position, content)
    for content, start, end in _find_all_commands(body, 'resumeGroupHeading'):
        tokens.append(('group', start, content))
    for content, start, end in _find_all_commands(body, 'resumeItem'):
        tokens.append(('item', start, content))
    for content, start, end in _find_all_commands(body, 'cvlistitem'):
        tokens.append(('item', start, content))
    # Sort by position in the source
    tokens.sort(key=lambda t: t[1])
    return [(kind, content) for kind, _, content in tokens]


def _parse_headingbf_experience(body: str) -> List[ExperienceBlock]:
    r"""Parse the headingBf/headingIt pattern used by some Overleaf templates.

    Structure:
        \headingBf{Employer Name}{Date Range}    <- employer header (bold)
          \headingIt{Project Title}{Date Range}  <- sub-project (italic)
          \begin{resume_list}
            \item bullet ...
          \end{resume_list}
          \headingIt{Next Project}{Date Range}
          ...

    Each \headingIt + its following bullet list becomes one ExperienceBlock
    with title=project, company=employer, dates=project dates.
    If there are bullets *before* the first \headingIt they go on the employer block.
    """
    # Regex to find \headingBf{name}{dates} and \headingIt{name}{dates}
    heading_re = re.compile(
        r'\\(headingBf|headingIt)\s*\{([^}]*)\}\s*\{([^}]*)\}',
        re.IGNORECASE,
    )
    # Collect bullet items from \begin{resume_list}...\end{resume_list} or \item lines
    item_re = re.compile(r'\\item\s+(.+?)(?=\\item|\\end\{resume_list\}|$)', re.DOTALL)

    headings: List[Tuple[int, str, str, str]] = []  # (pos, kind, name, dates)
    for m in heading_re.finditer(body):
        kind = m.group(1).lower()
        name = _strip_latex(m.group(2)).strip()
        dates = _strip_latex(m.group(3)).strip()
        headings.append((m.start(), kind, name, dates))

    if not headings:
        return []

    out: List[ExperienceBlock] = []
    current_employer = ""
    current_employer_dates = ""

    for i, (pos, kind, name, dates) in enumerate(headings):
        next_pos = headings[i + 1][0] if i + 1 < len(headings) else len(body)
        chunk = body[pos:next_pos]
        bullets: List[str] = []
        for bm in item_re.finditer(chunk):
            b = _strip_latex(bm.group(1)).strip()
            if b:
                bullets.append(b)

        if kind == 'headingbf':
            current_employer = name
            current_employer_dates = dates
            # Bullets directly under employer (before any headingIt) go on a top-level block
            if bullets:
                out.append(ExperienceBlock(
                    title=name, company="", dates=dates, bullets=bullets,
                ))
        else:  # headingit = sub-project
            out.append(ExperienceBlock(
                title=name,
                company=current_employer,
                dates=dates or current_employer_dates,
                bullets=bullets,
            ))

    return out


def _parse_experience(body: str) -> List[ExperienceBlock]:
    out: List[ExperienceBlock] = []

    # headingBf/headingIt pattern (used by some Overleaf templates)
    hbf_blocks = _parse_headingbf_experience(body)
    if hbf_blocks:
        return hbf_blocks

    # Find all \resumeSubheading and \cventry commands with brace-balanced extraction
    sub_positions: List[Tuple[int, str, str, str, str]] = []
    sub_re = re.compile(r'\\(?:resumeSubheading|cventry)\s*\{')
    for m in sub_re.finditer(body):
        pos = m.start()
        is_cventry = 'cventry' in m.group(0)
        cursor = m.end() - 1  # the '{'
        args: List[str] = []
        num_args = 6 if is_cventry else 4
        for _ in range(num_args):
            # Skip whitespace
            while cursor < len(body) and body[cursor] in ' \t\n\r':
                cursor += 1
            if cursor >= len(body) or body[cursor] != '{':
                break
            extracted = _extract_brace_arg(body, cursor)
            if extracted:
                args.append(extracted[0])
                cursor = extracted[1]
            else:
                break
        if len(args) == num_args:
            if is_cventry:
                loc = args[3].strip()
                if args[4].strip(): loc += f", {args[4].strip()}"
                if args[5].strip(): loc += f" - {args[5].strip()}"
                sub_positions.append((
                    pos,
                    _strip_latex(args[1]).strip(), # title
                    _strip_latex(args[0]).strip(), # dates
                    _strip_latex(args[2]).strip(), # company
                    _strip_latex(loc).strip()      # location
                ))
            else:
                sub_positions.append((pos, *(_strip_latex(a).strip() for a in args)))

    if not sub_positions:
        return _parse_experience_plain(body)

    for idx, (pos, title, dates, company, location) in enumerate(sub_positions):
        block = ExperienceBlock(
            title=title, dates=dates, company=company, location=location,
        )
        # Get the body between this subheading and the next
        next_pos = sub_positions[idx + 1][0] if idx + 1 < len(sub_positions) else len(body)
        chunk = body[pos:next_pos]

        # Tokenize the chunk for groups and items
        tokens = _tokenize_experience_body(chunk)
        current_group: Optional[ExperienceGroup] = None
        for kind, content in tokens:
            if kind == 'group':
                current_group = ExperienceGroup(label=_strip_latex(content))
                block.groups.append(current_group)
            elif kind == 'item':
                bullet = _strip_latex(content)
                if current_group is not None:
                    current_group.bullets.append(bullet)
                else:
                    block.bullets.append(bullet)
        out.append(block)
    return out


def _parse_experience_plain(body: str) -> List[ExperienceBlock]:
    """For non-Jake's-Resume templates - look for itemize blocks under headings."""
    plain = _strip_latex(body)
    blocks: List[ExperienceBlock] = []
    current: Optional[ExperienceBlock] = None
    date_re = re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{4}|"
        r"\d{1,2}/\d{4}|\d{4}\s*[-–—]\s*"
        r"(?:present|current|now|\d{4})",
        re.IGNORECASE,
    )
    for line in plain.splitlines():
        line = line.strip()
        if not line:
            continue
        is_bullet = line.startswith(("*", "-", "•", "–"))
        if is_bullet and current is not None:
            current.bullets.append(line.lstrip("*-•– ").strip())
            continue
        m = date_re.search(line)
        if m:
            if current:
                blocks.append(current)
            head = (line[:m.start()] + line[m.end():]).strip(" |-,")
            parts = re.split(r"\s+(?:at|@|\|)\s+|\s{2,}", head, maxsplit=1)
            title = parts[0]
            company = parts[1] if len(parts) > 1 else ""
            current = ExperienceBlock(
                title=title, company=company, dates=m.group(0).strip(),
            )
        elif current is None and len(line) < 140:
            current = ExperienceBlock(title="", company=line)
    if current:
        blocks.append(current)
    return blocks


# ----------------------------------------------------------------------
# Projects
# ----------------------------------------------------------------------
def _parse_projects(body: str) -> List[ProjectBlock]:
    out: List[ProjectBlock] = []
    # Re-use the experience layout if the user used \resumeSubheading
    exps = _parse_experience(body)
    for e in exps:
        out.append(ProjectBlock(
            name=e.title or e.company,
            tagline=e.company if e.title else "",
            dates=e.dates,
            bullets=list(e.bullets),
        ))
    if out:
        return out
    # \resumeProjectHeading{name}{dates} variant
    proj_re = re.compile(r"\\resumeProjectHeading\s*\{([^}]+)\}\s*\{([^}]+)\}", re.DOTALL)
    chunks = re.split(r"(?=\\resumeProjectHeading)", body)
    for chunk in chunks:
        m = proj_re.search(chunk)
        if not m:
            continue
        head = _strip_latex(m.group(1))
        dates = _strip_latex(m.group(2))
        name, _, tagline = head.partition(" - ")
        if not tagline:
            name, _, tagline = head.partition(" – ")
        if not tagline:
            name, _, tagline = head.partition(" — ")
        proj = ProjectBlock(name=name.strip(), tagline=tagline.strip(), dates=dates.strip())
        body_part = chunk[m.end():]
        for b in re.finditer(r"\\resumeItem\s*\{(.+?)\}\s*(?=\\)", body_part, re.DOTALL):
            proj.bullets.append(_strip_latex(b.group(1)))
        out.append(proj)
    if out:
        return out
    # Plain-text fallback
    plain = _strip_latex(body)
    current: Optional[ProjectBlock] = None
    for raw in plain.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("*", "-", "•")):
            if current is not None:
                current.bullets.append(line.lstrip("*-• ").strip())
        else:
            if current:
                out.append(current)
            current = ProjectBlock(name=line)
    if current:
        out.append(current)
    return out


# ----------------------------------------------------------------------
# Education
# ----------------------------------------------------------------------
def _parse_education(body: str) -> List[EducationBlock]:
    blocks: List[EducationBlock] = []

    # 1. ModernCV cvcolumns format
    cvcols_re = re.compile(r'\\cvcolumn\*?(?:\s*\[[^\]]*\])?\s*\{')
    col_cells = []
    for m in cvcols_re.finditer(body):
        cursor = m.end() - 1
        args = []
        for _ in range(2):
            while cursor < len(body) and body[cursor] in ' \t\n\r':
                cursor += 1
            if cursor >= len(body) or body[cursor] != '{':
                break
            extracted = _extract_brace_arg(body, cursor)
            if extracted:
                args.append(extracted[0])
                cursor = extracted[1]
            else:
                break
        if len(args) == 2:
            content = args[1]
            cells = [_strip_latex(c).strip() for c, _, _ in _find_all_commands(content, 'cvcolumncell')]
            if not cells:
                cells = [line.strip() for line in _strip_latex(content).splitlines() if line.strip()]
            col_cells.append(cells)
            
    if col_cells:
        num_rows = max(len(c) for c in col_cells)
        for i in range(num_rows):
            degree = col_cells[0][i] if i < len(col_cells[0]) else ""
            inst = col_cells[1][i] if len(col_cells) > 1 and i < len(col_cells[1]) else ""
            dates = col_cells[2][i] if len(col_cells) > 2 and i < len(col_cells[2]) else ""
            blocks.append(EducationBlock(degree=degree, institution=inst, dates=dates))
        return blocks

    # 2. ModernCV \cventry format
    # Education can also be written using \cventry
    sub_re = re.compile(r'\\cventry\s*\{')
    sub_positions = []
    for m in sub_re.finditer(body):
        pos = m.start()
        cursor = m.end() - 1
        args: List[str] = []
        for _ in range(6):
            while cursor < len(body) and body[cursor] in ' \t\n\r':
                cursor += 1
            if cursor >= len(body) or body[cursor] != '{':
                break
            extracted = _extract_brace_arg(body, cursor)
            if extracted:
                args.append(extracted[0])
                cursor = extracted[1]
            else:
                break
        if len(args) == 6:
            loc = args[3].strip()
            if args[4].strip(): loc += f", {args[4].strip()}"
            sub_positions.append((pos, _strip_latex(args[1]).strip(), _strip_latex(args[0]).strip(), _strip_latex(args[2]).strip(), _strip_latex(loc).strip()))
    
    if sub_positions:
        for pos, degree, dates, institution, location in sub_positions:
            blocks.append(EducationBlock(degree=degree, dates=dates, institution=institution, location=location))
        return blocks

    # 3. Jake's Resume format
    pattern = re.compile(
        r"\\textbf\s*\{([^}]+)\}\s*&\s*\\textit\s*\{([^}]+)\}\s*\\\\"
        r"\s*\{?\\small\s*([^&}]+?)\}?\s*&\s*\{?\\small\s*([^&}\\]+?)\}?\s*\\\\",
        re.DOTALL,
    )
    matches = list(pattern.finditer(body))
    if not matches:
        # Looser fallback
        for m in re.finditer(
            r"\\textbf\s*\{([^}]+)\}.*?\\textit\s*\{([^}]+)\}.*?\\small\s*([^}]+)\}",
            body, re.DOTALL,
        ):
            degree, dates, institution = (_strip_latex(s) for s in m.groups())
            blocks.append(EducationBlock(
                degree=degree, dates=dates, institution=institution,
            ))
        if not blocks:
            return _parse_education_plain(body)
        return blocks
    italic_re = re.compile(r"\\textit\s*\{((?:[^{}]|\{[^}]*\})*)\}")
    for i, m in enumerate(matches):
        degree, dates, institution, location = (
            _strip_latex(s).strip() for s in m.groups()
        )
        block = EducationBlock(
            degree=degree, dates=dates,
            institution=institution, location=location,
        )
        chunk_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[m.end():chunk_end]
        for im in italic_re.finditer(chunk):
            inner = _strip_latex(im.group(1)).strip()
            low = inner.lower()
            if low.startswith(("relevant coursework", "coursework")):
                _, _, txt = inner.partition(":")
                block.coursework = txt.strip() or inner
            elif low.startswith(("dissertation", "thesis")):
                _, _, txt = inner.partition(":")
                block.dissertation = txt.strip() or inner
        blocks.append(block)
    return blocks


def _parse_education_plain(body: str) -> List[EducationBlock]:
    plain = _strip_latex(body)
    blocks: List[EducationBlock] = []
    deg_re = re.compile(
        r"\b(M\.?Tech|B\.?Tech|M\.?Sc|B\.?Sc|M\.?S\b|B\.?S\b|MBA|MBBS|"
        r"Ph\.?D|Bachelor|Master|Doctorate|Diploma|Associate)\b",
        re.IGNORECASE,
    )
    for raw in plain.splitlines():
        line = raw.strip()
        if not line:
            continue
        if deg_re.search(line):
            parts = re.split(r"\s{2,}|\|", line)
            blocks.append(EducationBlock(
                degree=parts[0].strip(),
                institution=parts[1].strip() if len(parts) > 1 else "",
                dates=parts[-1].strip() if len(parts) > 2 else "",
            ))
        elif blocks and not blocks[-1].institution:
            parts = re.split(r"\s{2,}|\|", line)
            blocks[-1].institution = parts[0].strip()
            if len(parts) > 1:
                blocks[-1].location = parts[-1].strip()
    return blocks


# ----------------------------------------------------------------------
# Certifications + publications + achievements
# ----------------------------------------------------------------------
def _parse_certifications(body: str) -> List[CertificationItem]:
    plain = _strip_latex(body)
    out: List[CertificationItem] = []
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    for raw in plain.splitlines():
        line = raw.lstrip("*-• ").strip()
        if not line:
            continue
        year = ""
        ym = year_re.search(line)
        if ym:
            year = ym.group(0)
            line = line.replace(year, "").strip(" ,()-")
        parts = re.split(r"\s+[–—-]\s+|\s+\|\s+|,\s+", line, maxsplit=1)
        out.append(CertificationItem(
            name=parts[0].strip(),
            issuer=parts[1].strip() if len(parts) > 1 else "",
            year=year,
        ))
    return out


def _parse_publications(body: str) -> List[PublicationItem]:
    plain = _strip_latex(body)
    out: List[PublicationItem] = []
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    for raw in plain.splitlines():
        line = raw.lstrip("*-• ").strip()
        if not line:
            continue
        year = ""
        ym = year_re.search(line)
        if ym:
            year = ym.group(0)
        parts = re.split(r"\s+[–—\-]\s+|,\s+", line, maxsplit=1)
        out.append(PublicationItem(
            title=parts[0].strip(),
            venue=parts[1].strip() if len(parts) > 1 else "",
            year=year,
        ))
    return out


def _parse_achievements(body: str) -> List[AchievementItem]:
    out: List[AchievementItem] = []
    # \achieveRow{title}{description}{year-or-org}
    for m in re.finditer(
        r"\\achieveRow\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}", body,
    ):
        title, desc, yoo = (_strip_latex(s) for s in m.groups())
        out.append(AchievementItem(title=title, description=desc, year_or_org=yoo))
    if out:
        return out
    # Plain-text fallback - one row per non-empty line
    plain = _strip_latex(body)
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    for raw in plain.splitlines():
        line = raw.lstrip("*-• ").strip()
        if not line:
            continue
        year = ""
        ym = year_re.search(line)
        if ym:
            year = ym.group(0)
            line = (line[:ym.start()] + line[ym.end():]).strip(" ,()-")
        parts = re.split(r"\s+\|\s+|\s+[–—\-]\s+", line, maxsplit=1)
        title = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        out.append(AchievementItem(title=title, description=desc, year_or_org=year))
    return out


def _parse_summary(body: str) -> str:
    return _strip_latex(body).strip()


def _parse_extra(title: str, body: str) -> ExtraSection:
    plain = _strip_latex(body)
    items: List[str] = []
    for raw in plain.splitlines():
        line = raw.lstrip("*-• ").strip()
        if line:
            items.append(line)
    return ExtraSection(title=title, body=plain, items=items)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def parse_tex_to_ir(path: Path | str) -> ResumeIR:
    """Parse a .tex resume file into a fully populated ResumeIR."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return parse_tex_string(text)


def parse_tex_string(text: str) -> ResumeIR:
    text = _strip_tex_comments(text)

    header = HeaderInfo(
        name=_extract_name(text),
        headline=_extract_headline(text),
    )
    links, location = _extract_links(text)
    header.links = links
    if location:
        header.location = location

    ir = ResumeIR(header=header)

    sections = _split_sections(text)
    seen_kinds: set[str] = set()
    actual_order: List[str] = []

    for raw_title, body in sections:
        kind = _normalize_section(raw_title)
        if not kind:
            ir.extras.append(_parse_extra(raw_title, body))
            if "extras" not in actual_order:
                actual_order.append("extras")
            continue
        if kind in seen_kinds:
            # Same canonical section appearing twice - merge as extra
            ir.extras.append(_parse_extra(raw_title, body))
            if "extras" not in actual_order:
                actual_order.append("extras")
            continue
        seen_kinds.add(kind)
        actual_order.append(kind)
        if kind == "summary":
            ir.summary = _parse_summary(body)
        elif kind == "skills":
            ir.skills = _parse_skills(body)
        elif kind == "experience":
            ir.experience = _parse_experience(body)
        elif kind == "projects":
            ir.projects = _parse_projects(body)
        elif kind == "education":
            ir.education = _parse_education(body)
        elif kind == "certifications":
            ir.certifications = _parse_certifications(body)
        elif kind == "achievements":
            ir.achievements = _parse_achievements(body)
        elif kind == "publications":
            ir.publications = _parse_publications(body)

    if actual_order:
        ir.section_order = actual_order

    return ir
