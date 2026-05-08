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
    "underline",
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
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------
# Header extraction
# ----------------------------------------------------------------------
_NAME_PATTERNS = [
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
    r"\\section\*?\{(?P<title>[^}]+)\}(?P<body>.*?)(?=\\section\*?\{|\\end\{document\})",
    re.DOTALL,
)


def _split_sections(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in _SECTION_BODY_RE.finditer(text):
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


# ----------------------------------------------------------------------
# Experience
# ----------------------------------------------------------------------
_SUB_RE = re.compile(
    r"\\resumeSubheading\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}"
)
_TOKEN_RE = re.compile(
    r"\\resumeGroupHeading\s*\{([^}]+)\}|\\resumeItem\s*\{(.+?)\}"
    r"(?=\s*(?:\\resumeItem|\\resumeGroupHeading|\\resumeItemListEnd|\\resumeSubheading"
    r"|\\resumeSubHeadingListEnd|\\employerSeparator|$))",
    re.DOTALL,
)


def _parse_experience(body: str) -> List[ExperienceBlock]:
    out: List[ExperienceBlock] = []
    chunks = re.split(r"(?=\\resumeSubheading)", body)
    for chunk in chunks:
        m = _SUB_RE.search(chunk)
        if not m:
            continue
        title, dates, company, location = (_strip_latex(s).strip() for s in m.groups())
        block = ExperienceBlock(
            title=title, dates=dates, company=company, location=location,
        )
        body_part = chunk[m.end():]
        current_group: Optional[ExperienceGroup] = None
        for tok in _TOKEN_RE.finditer(body_part):
            grp_label = tok.group(1)
            item_body = tok.group(2)
            if grp_label is not None:
                current_group = ExperienceGroup(label=_strip_latex(grp_label))
                block.groups.append(current_group)
            elif item_body is not None:
                bullet = _strip_latex(item_body)
                if current_group is not None:
                    current_group.bullets.append(bullet)
                else:
                    block.bullets.append(bullet)
        # Greedy fallback if the lookahead missed everything
        if not block.bullets and not block.groups:
            for b in re.finditer(r"\\resumeItem\s*\{(.+?)\}\s*(?=\\)", body_part, re.DOTALL):
                block.bullets.append(_strip_latex(b.group(1)))
        out.append(block)
    if out:
        return out
    # Plain-text fallback - no \resumeSubheading template
    return _parse_experience_plain(body)


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
    for raw_title, body in sections:
        kind = _normalize_section(raw_title)
        if not kind:
            ir.extras.append(_parse_extra(raw_title, body))
            continue
        if kind in seen_kinds:
            # Same canonical section appearing twice - merge as extra
            ir.extras.append(_parse_extra(raw_title, body))
            continue
        seen_kinds.add(kind)
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

    return ir
