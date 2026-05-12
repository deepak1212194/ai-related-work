"""
ExtractorAgent - LLM-assisted post-processing of the heuristic .tex parse.

The .tex parser does the heavy lifting deterministically; this agent's job
is to (a) repair common parse errors, (b) detect missed sections, (c) lift
a headline when the input has none, (d) recover LinkedIn/GitHub links that
the regex missed, and (e) extract certifications from unparsed sections.

Falls back to the heuristic parse on any LLM failure.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..core.context_budget import estimate_tokens
from ..core.ir import (
    AchievementItem, CertificationItem, ContactLink, EducationBlock,
    ExperienceBlock, ExperienceGroup, ExtraSection, HeaderInfo,
    ProjectBlock, ResumeIR, SkillBucket,
)
from .base import Agent, extract_json

log = logging.getLogger(__name__)

_VALID_LINK_KINDS = {
    "email", "phone", "linkedin", "github", "website",
    "twitter", "scholar", "portfolio", "other",
}


def _needs_link_repair(ir: ResumeIR, raw_tex: str) -> bool:
    """True when LinkedIn or GitHub appears in raw .tex but not in parsed links."""
    existing = {lk.kind for lk in ir.header.links}
    raw_lower = raw_tex.lower()
    return (
        ("linkedin.com" in raw_lower and "linkedin" not in existing)
        or ("github.com" in raw_lower and "github" not in existing)
    )


def _needs_cert_repair(ir: ResumeIR, raw_tex: str) -> bool:
    """True when a Certifications/Licenses section exists in .tex but IR has none."""
    if ir.certifications:
        return False
    return bool(re.search(
        r"\\(?:section|tinysection)\s*\{[^}]*(certif|licens|credential)[^}]*\}",
        raw_tex, re.IGNORECASE,
    ))


class ExtractorAgent(Agent):
    @staticmethod
    def _compact_tex_excerpt(raw_tex: str, max_chars: int = 2400) -> str:
        """Return a high-signal subset of tex content for repair calls."""
        if len(raw_tex) <= max_chars:
            return raw_tex
        sections = []
        for marker in (r"\begin{center}", r"\section", r"\cvsection", r"\tinysection", r"\documentTitle{", r"\name{"):
            idx = raw_tex.find(marker)
            if idx >= 0:
                sections.append(raw_tex[idx: idx + 900])
        merged = "\n".join(s for s in sections if s)
        if merged:
            return merged[:max_chars]
        return raw_tex[:max_chars]

    def repair(self, ir: ResumeIR, raw_tex_excerpt: str) -> ResumeIR:
        """
        Best-effort LLM-assisted repair of the heuristic parse.

        Covers: headline, experience titles/companies/dates, education
        institution/location swaps, LinkedIn/GitHub links missed by regex,
        certifications in unparsed sections, and any missed extra sections.
        Falls back to the input IR unchanged on any error.
        """
        needs_repair = (
            not ir.header.headline
            or any(not e.title and e.bullets for e in ir.experience)
            or any(not ed.institution for ed in ir.education)
            or _needs_link_repair(ir, raw_tex_excerpt)
            or _needs_cert_repair(ir, raw_tex_excerpt)
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
                f"RAW_TEX_COMPACT:\n```\n{self._compact_tex_excerpt(raw_tex_excerpt)}\n```\n\n"
                "Return a JSON object with these top-level keys (only include those you want to change):\n"
                "- headline (string)\n"
                "- experience_repairs: list of {index, title?, company?, dates?}\n"
                "- education_repairs: list of {index, institution?, location?}\n"
                "- link_repairs: list of {kind, label, url} — links in raw .tex missing from header\n"
                "- certification_repairs: list of {name, issuer, year} — certs found in raw .tex but not parsed\n"
                "- missed_sections: list of {title, items: [string, ...]}\n"
                "\nRules: minimal edits only, no invention, no extra keys.\n"
            )
            if estimate_tokens(sys_prompt + user) > 3600:
                user = (
                    f"CURRENT_PARSE (JSON):\n{ir.model_dump_json()[:1800]}\n\n"
                    f"RAW_TEX_COMPACT:\n```\n{self._compact_tex_excerpt(raw_tex_excerpt, max_chars=1200)}\n```\n\n"
                    "Return JSON with only keys that require fixes."
                )
            raw = self.llm.complete(sys_prompt, user, max_tokens=1000, temperature=0.1)
        except Exception as e:                              # noqa: BLE001
            log.warning("[Extractor] LLM repair failed: %s", e)
            return ir

        data = extract_json(raw)
        if not isinstance(data, dict):
            return ir

        # headline
        if isinstance(data.get("headline"), str) and data["headline"].strip() and not ir.header.headline:
            ir.header.headline = data["headline"].strip()[:160]

        # experience
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

        # education
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

        # link repairs — only add kinds that are missing
        existing_kinds = {lk.kind for lk in ir.header.links}
        for lr in data.get("link_repairs", []) or []:
            if not isinstance(lr, dict):
                continue
            kind  = str(lr.get("kind", "other")).strip().lower()
            url   = str(lr.get("url",   "")).strip()[:300]
            label = str(lr.get("label", "")).strip()[:80]
            if not url or kind not in _VALID_LINK_KINDS:
                continue
            if kind in existing_kinds:
                continue
            ir.header.links.append(ContactLink(kind=kind, label=label or kind.title(), url=url))
            existing_kinds.add(kind)

        # certification repairs — only if currently empty
        if not ir.certifications:
            for cr in data.get("certification_repairs", []) or []:
                if not isinstance(cr, dict):
                    continue
                name = str(cr.get("name", "")).strip()[:200]
                if not name:
                    continue
                ir.certifications.append(CertificationItem(
                    name=name,
                    issuer=str(cr.get("issuer", "")).strip()[:100],
                    year=str(cr.get("year", "")).strip()[:10],
                    description=str(cr.get("description", "")).strip()[:300],
                ))

        # missed sections
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

    # ------------------------------------------------------------------
    # LLM-first full extraction (primary parse path)
    # ------------------------------------------------------------------

    def extract_full(self, raw_tex: str) -> Optional[ResumeIR]:
        """LLM-first primary extraction for any .tex template.

        Sends the complete raw .tex to the LLM with an explicit JSON schema
        that maps to every field the output template needs.  Returns a
        populated ResumeIR on success, None on any failure so the pipeline
        can fall back to the regex parser.
        """
        # 12 000 chars ≈ 3 000 tokens — covers any real resume and is well
        # within the context windows of all supported models (Llama 4 Scout:
        # 131K, llama-3.1-8b-instant: 128K, llama-3.3-70b-versatile: 128K).
        tex_input = raw_tex[:12000] if len(raw_tex) > 12000 else raw_tex

        sys_prompt = (
            "You are an expert resume parser. Extract ALL information from the "
            "LaTeX resume into a structured JSON object.\n"
            "The LaTeX may use ANY custom commands — \\documentTitle, \\tinysection, "
            "\\heading, \\begin{resume_list}, \\resumeSubheading, \\section, etc. "
            "Look for content regardless of the command names used.\n\n"
            "Return ONLY a valid JSON object with EXACTLY these keys:\n"
            "{\n"
            '  "name": "Full name of the person (look at the very top of the document)",\n'
            '  "headline": "Professional title or tagline beneath the name",\n'
            '  "email": "email@domain.com",\n'
            '  "phone": "+1 234 567 8900",\n'
            '  "linkedin": "https://linkedin.com/in/handle",\n'
            '  "github": "https://github.com/handle",\n'
            '  "location": "City, Country",\n'
            '  "summary": "Full text of the professional summary / objective paragraph",\n'
            '  "skills": [\n'
            '    {"name": "Languages", "items": ["Python", "Java", "SQL"]},\n'
            '    {"name": "Frameworks", "items": ["TensorFlow", "PyTorch"]}\n'
            '  ],\n'
            '  "experience": [\n'
            '    {\n'
            '      "title": "Senior Data Scientist",\n'
            '      "company": "Acme Corp",\n'
            '      "dates": "Jan 2022 – Present",\n'
            '      "location": "New York, NY",\n'
            '      "summary_line": "Optional one-liner about the team or product",\n'
            '      "bullets": ["Achievement 1 using tech X", "Led initiative Y..."]\n'
            '    }\n'
            '  ],\n'
            '  "projects": [\n'
            '    {"name": "Project Name", "tagline": "Tech / Description", '
            '"dates": "2023", "bullets": ["..."]}\n'
            '  ],\n'
            '  "education": [\n'
            '    {"degree": "B.Tech CS", "institution": "IIT Bombay", '
            '"dates": "2018-2022", "location": "Mumbai", "gpa": ""}\n'
            '  ],\n'
            '  "certifications": [\n'
            '    {"name": "AWS SAA", "issuer": "Amazon", "year": "2023", '
            '"description": "Credential ID: ABC123 | Cloud architecture, S3, EC2, IAM"}\n'
            '  ],\n'
            '  "achievements": [\n'
            '    {"title": "Award Name", "description": "Won X for Y", '
            '"year_or_org": "2022"}\n'
            '  ]\n'
            "}\n\n"
            "CRITICAL RULES:\n"
            "1. Find the person's REAL FULL NAME — it may be in \\documentTitle{Name}{}, "
            "   \\Huge{\\textbf{Name}}, \\name{Name}, or a large-font line at the top.\n"
            "2. Extract EVERY experience entry and ALL bullet points verbatim.\n"
            "3. If experience entries contain sub-projects (e.g. 'Project – Polaris'), "
            "   group their bullets under the same parent company entry; add the project "
            "   name as a prefix in the bullet (e.g. '[Polaris] Built forecasting model...').\n"
            "4. Use \"\" for missing strings, [] for missing arrays.\n"
            "5. Do NOT invent or rephrase — extract exactly what is written."
        )

        user = f"Parse this LaTeX resume and return the JSON:\n```\n{tex_input}\n```"

        try:
            raw = self.llm.complete(sys_prompt, user, max_tokens=3000, temperature=0.05)
        except Exception as e:
            log.warning("[Extractor] extract_full LLM call failed: %s", e)
            return None

        data = extract_json(raw)
        if not isinstance(data, dict):
            log.warning("[Extractor] extract_full: non-dict response (%r…)", raw[:200])
            return None

        try:
            ir = self._build_ir_from_llm(data)
        except Exception as e:
            log.warning("[Extractor] extract_full: build_ir_from_llm failed: %s", e)
            return None

        if not ir.header.name:
            log.warning("[Extractor] extract_full: name not found in LLM response")
            return None

        log.info(
            "[Extractor] extract_full OK: name=%r exp=%d skills=%d edu=%d cert=%d",
            ir.header.name, len(ir.experience), len(ir.skills),
            len(ir.education), len(ir.certifications),
        )
        return ir

    def _build_ir_from_llm(self, data: dict) -> ResumeIR:
        """Convert the LLM structured JSON response into a ResumeIR object."""
        links: List[ContactLink] = []

        if data.get("email"):
            email = str(data["email"]).strip()
            links.append(ContactLink(
                kind="email", label=email, url=f"mailto:{email}", icon="faEnvelope",
            ))
        if data.get("phone"):
            ph = str(data["phone"]).strip()
            digits = re.sub(r"[^+0-9]", "", ph)
            links.append(ContactLink(
                kind="phone", label=ph, url=f"tel:{digits}", icon="faPhone",
            ))
        if data.get("linkedin"):
            url = str(data["linkedin"]).strip()
            if not url.startswith("http"):
                url = "https://" + url
            label = url.split("linkedin.com/")[-1].rstrip("/")
            links.append(ContactLink(
                kind="linkedin", label=label, url=url, icon="faLinkedin",
            ))
        if data.get("github"):
            url = str(data["github"]).strip()
            if not url.startswith("http"):
                url = "https://" + url
            label = url.split("github.com/")[-1].rstrip("/")
            links.append(ContactLink(
                kind="github", label=label, url=url, icon="faGithub",
            ))
        if data.get("location"):
            loc = str(data["location"]).strip()
            links.append(ContactLink(
                kind="location", label=loc, url="", icon="faMapMarker",
            ))

        header = HeaderInfo(
            name=str(data.get("name") or "").strip()[:150],
            headline=str(data.get("headline") or "").strip()[:200],
            links=links,
        )

        skills: List[SkillBucket] = []
        for s in data.get("skills") or []:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "").strip()
            items = [str(i).strip() for i in (s.get("items") or []) if str(i).strip()]
            if name:
                skills.append(SkillBucket(name=name, items=items))

        experience: List[ExperienceBlock] = []
        for e in data.get("experience") or []:
            if not isinstance(e, dict):
                continue
            bullets = [str(b).strip() for b in (e.get("bullets") or []) if str(b).strip()]
            experience.append(ExperienceBlock(
                title=str(e.get("title") or "").strip()[:150],
                company=str(e.get("company") or "").strip()[:150],
                dates=str(e.get("dates") or "").strip()[:80],
                location=str(e.get("location") or "").strip()[:120],
                summary_line=str(e.get("summary_line") or "").strip()[:200],
                bullets=bullets,
            ))

        projects: List[ProjectBlock] = []
        for p in data.get("projects") or []:
            if not isinstance(p, dict):
                continue
            bullets = [str(b).strip() for b in (p.get("bullets") or []) if str(b).strip()]
            projects.append(ProjectBlock(
                name=str(p.get("name") or "").strip()[:150],
                tagline=str(p.get("tagline") or "").strip()[:200],
                dates=str(p.get("dates") or "").strip()[:80],
                bullets=bullets,
            ))

        education: List[EducationBlock] = []
        for e in data.get("education") or []:
            if not isinstance(e, dict):
                continue
            education.append(EducationBlock(
                degree=str(e.get("degree") or "").strip()[:200],
                institution=str(e.get("institution") or "").strip()[:200],
                dates=str(e.get("dates") or "").strip()[:80],
                location=str(e.get("location") or "").strip()[:120],
                gpa=str(e.get("gpa") or "").strip()[:20],
            ))

        certifications: List[CertificationItem] = []
        for c in data.get("certifications") or []:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()[:200]
            if name:
                certifications.append(CertificationItem(
                    name=name,
                    issuer=str(c.get("issuer") or "").strip()[:100],
                    year=str(c.get("year") or "").strip()[:10],
                    description=str(c.get("description") or "").strip()[:300],
                ))

        achievements: List[AchievementItem] = []
        for a in data.get("achievements") or []:
            if not isinstance(a, dict):
                continue
            title = str(a.get("title") or "").strip()[:200]
            if title:
                achievements.append(AchievementItem(
                    title=title,
                    description=str(a.get("description") or "").strip()[:500],
                    year_or_org=str(a.get("year_or_org") or "").strip()[:50],
                ))

        return ResumeIR(
            header=header,
            summary=str(data.get("summary") or "").strip(),
            skills=skills,
            experience=experience,
            projects=projects,
            education=education,
            certifications=certifications,
            achievements=achievements,
        )
