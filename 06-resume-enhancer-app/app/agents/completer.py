"""
CompleterAgent - fills required-field placeholders for missing data.

Keeps the agent's output deterministic: if the input has all required
fields, this is a no-op. If it doesn't, we insert obviously-flagged
placeholders rather than calling the LLM (the LLM can't invent without
breaching the honesty rule, so a deterministic placeholder is safer).
"""

from __future__ import annotations

import logging
from typing import List

from ..core.ir import (
    AchievementItem, CertificationItem, ContactLink, EducationBlock,
    ExperienceBlock, HeaderInfo, ResumeIR, SkillBucket,
)
from .base import Agent

log = logging.getLogger(__name__)


PLACEHOLDER_NAME = "[YOUR FULL NAME]"
PLACEHOLDER_HEADLINE = "[ONE-LINE HEADLINE - role + domain + signature stack]"


def _ensure_name(ir: ResumeIR, completed: List[str]) -> None:
    if not ir.header.name:
        ir.header.name = PLACEHOLDER_NAME
        completed.append("header.name")


def _ensure_headline(ir: ResumeIR, completed: List[str]) -> None:
    if not ir.header.headline:
        # Derive headline from the most senior experience title available
        if ir.experience and ir.experience[0].title:
            blk = ir.experience[0]
            headline = blk.title
            # Add company domain context if available
            if blk.company:
                headline = f"{blk.title} at {blk.company}"
            ir.header.headline = headline[:160]
            completed.append("header.headline (derived)")
        elif ir.summary:
            # Extract first clause of summary (up to the first period or 100 chars)
            first_clause = ir.summary.split(".")[0].strip()
            if first_clause and len(first_clause) > 10:
                ir.header.headline = first_clause[:160]
                completed.append("header.headline (derived from summary)")
        else:
            ir.header.headline = PLACEHOLDER_HEADLINE
            completed.append("header.headline (placeholder)")


def _ensure_links(ir: ResumeIR, completed: List[str]) -> None:
    have = {l.kind for l in ir.header.links}
    if "email" not in have:
        ir.header.links.append(ContactLink(
            kind="email", label="[your.email@domain.com]", url="mailto:[your.email@domain.com]",
            icon="faEnvelope", placeholder=True,
        ))
        completed.append("header.links.email")
    if "linkedin" not in have:
        ir.header.links.append(ContactLink(
            kind="linkedin", label="linkedin.com/in/[your-handle]",
            url="https://linkedin.com/in/[your-handle]",
            icon="faLinkedin", placeholder=True,
        ))
        completed.append("header.links.linkedin")
    if "github" not in have:
        ir.header.links.append(ContactLink(
            kind="github", label="github.com/[your-handle]",
            url="https://github.com/[your-handle]",
            icon="faGithub", placeholder=True,
        ))
        completed.append("header.links.github")


def _ensure_summary(ir: ResumeIR, completed: List[str]) -> None:
    if not ir.summary:
        ir.summary = (
            "[ADD 2-3 LINE SUMMARY: years of experience, primary domain, top "
            "technologies, and one signature outcome.]"
        )
        completed.append("summary (placeholder)")


def _ensure_skills(ir: ResumeIR, completed: List[str]) -> None:
    if not ir.skills:
        ir.skills = [
            SkillBucket(name="Languages", items=["[language 1]", "[language 2]"], placeholder=True),
            SkillBucket(name="Frameworks", items=["[framework 1]", "[framework 2]"], placeholder=True),
            SkillBucket(name="Cloud & DevOps", items=["[cloud]", "[orchestration]"], placeholder=True),
            SkillBucket(name="Data & Backend", items=["[database]", "[api framework]"], placeholder=True),
        ]
        completed.append("skills (placeholder buckets)")


def _ensure_experience(ir: ResumeIR, completed: List[str]) -> None:
    if not ir.experience:
        ir.experience = [
            ExperienceBlock(
                title="[ROLE TITLE]",
                company="[COMPANY NAME]",
                location="[CITY, COUNTRY]",
                dates="[START - END]",
                summary_line="[ONE-LINE CONTEXT - product or team]",
                bullets=[
                    "Architected [system or scope] using [stack] - delivered [impact].",
                    "Owned [system or feature] from [start] to [milestone], integrating [tools].",
                    "Designed [component or pipeline] that [outcome] using [technique].",
                ],
                placeholder=True,
            )
        ]
        completed.append("experience (placeholder block)")


def _ensure_education(ir: ResumeIR, completed: List[str]) -> None:
    if not ir.education:
        ir.education = [
            EducationBlock(
                degree="[DEGREE NAME]",
                institution="[INSTITUTION NAME]",
                location="[CITY, COUNTRY]",
                dates="[YYYY - YYYY]",
                placeholder=True,
            )
        ]
        completed.append("education (placeholder block)")


class CompleterAgent(Agent):
    def fill(self, ir: ResumeIR) -> ResumeIR:
        """Return the IR with required-field placeholders filled in.

        Education is NOT auto-filled: if the source resume has no
        education section we do not invent one.  Every other required
        field gets a visible [PLACEHOLDER] token.
        """
        completed: List[str] = []
        _ensure_name(ir, completed)
        _ensure_links(ir, completed)
        _ensure_headline(ir, completed)
        _ensure_summary(ir, completed)
        _ensure_skills(ir, completed)
        _ensure_experience(ir, completed)
        # Education: only patch partial blocks, never create from scratch.
        for ed in ir.education:
            if not ed.degree and not ed.placeholder:
                ed.degree = "[DEGREE NAME]"
                ed.placeholder = True
                completed.append("education.degree (placeholder)")
        if completed:
            log.info("[Completer] filled %d placeholder field(s): %s",
                     len(completed), ", ".join(completed))
            ir.completed_fields = list(set(ir.completed_fields + completed))
        return ir
