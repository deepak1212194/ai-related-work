"""
ir.py - Resume Intermediate Representation (IR).

Every agent reads and writes this single typed structure. Parsing,
extraction, enhancement, and rendering all interoperate through ResumeIR.

Design principles:
- Every field has a sensible empty default so partial extraction never
  crashes the renderer.
- `placeholder` flags mark fields the Completer agent filled in because
  they were missing from the input but required by the template.
- Provenance: every enhanced section keeps `original` and `enhanced`
  side-by-side plus the rubric trace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Identity / contact
# ----------------------------------------------------------------------
class ContactLink(BaseModel):
    kind: Literal[
        "email", "phone", "linkedin", "github", "website",
        "location", "twitter", "scholar", "portfolio", "other",
    ] = "other"
    label: str = ""
    url: str = ""
    icon: str = ""
    placeholder: bool = False


class HeaderInfo(BaseModel):
    name: str = ""
    headline: str = ""
    location: str = ""
    links: List[ContactLink] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Core sections
# ----------------------------------------------------------------------
class SkillBucket(BaseModel):
    """One row of the Technical Skills section."""
    name: str
    items: List[str] = Field(default_factory=list)
    placeholder: bool = False


class ExperienceGroup(BaseModel):
    label: str = ""
    bullets: List[str] = Field(default_factory=list)


class ExperienceBlock(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    dates: str = ""
    summary_line: str = ""
    bullets: List[str] = Field(default_factory=list)
    groups: List[ExperienceGroup] = Field(default_factory=list)
    placeholder: bool = False


class EducationBlock(BaseModel):
    degree: str = ""
    institution: str = ""
    location: str = ""
    dates: str = ""
    coursework: str = ""
    dissertation: str = ""
    gpa: str = ""
    honors: str = ""
    placeholder: bool = False


class ProjectBlock(BaseModel):
    name: str = ""
    tagline: str = ""
    link: str = ""
    dates: str = ""
    stack: str = ""
    bullets: List[str] = Field(default_factory=list)
    placeholder: bool = False


class CertificationItem(BaseModel):
    name: str = ""
    issuer: str = ""
    year: str = ""
    placeholder: bool = False


class AchievementItem(BaseModel):
    """Row used by the v2 \\achieveRow{title}{description}{year-or-org} macro."""
    title: str = ""
    description: str = ""
    year_or_org: str = ""
    placeholder: bool = False


class PublicationItem(BaseModel):
    title: str = ""
    venue: str = ""
    year: str = ""
    link: str = ""
    placeholder: bool = False


class ExtraSection(BaseModel):
    """Catch-all so unrecognised input sections survive the round-trip."""
    title: str
    body: str = ""
    items: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Top-level IR
# ----------------------------------------------------------------------
class ResumeIR(BaseModel):
    header: HeaderInfo = Field(default_factory=HeaderInfo)
    summary: str = ""
    skills: List[SkillBucket] = Field(default_factory=list)
    experience: List[ExperienceBlock] = Field(default_factory=list)
    projects: List[ProjectBlock] = Field(default_factory=list)
    education: List[EducationBlock] = Field(default_factory=list)
    certifications: List[CertificationItem] = Field(default_factory=list)
    achievements: List[AchievementItem] = Field(default_factory=list)
    publications: List[PublicationItem] = Field(default_factory=list)
    extras: List[ExtraSection] = Field(default_factory=list)

    # Order in which sections render. Defaults to the v2 layout.
    section_order: List[str] = Field(default_factory=lambda: [
        "summary", "skills", "experience", "projects",
        "education", "certifications", "achievements",
        "publications", "extras",
    ])

    # Anything the Completer filled in.
    completed_fields: List[str] = Field(default_factory=list)

    def has_required(self) -> Dict[str, bool]:
        return {
            "name": bool(self.header.name),
            "summary": bool(self.summary),
            "skills": bool(self.skills),
            "experience": bool(self.experience),
            "education": bool(self.education),
        }

    def text_blob(self) -> str:
        """Concatenated plain text - used for ATS keyword scoring."""
        parts: List[str] = []
        h = self.header
        parts.append(h.name)
        parts.append(h.headline)
        parts.append(h.location)
        for ln in h.links:
            parts.append(f"{ln.label} {ln.url}")
        parts.append(self.summary)
        for s in self.skills:
            parts.append(f"{s.name}: {', '.join(s.items)}")
        for e in self.experience:
            parts.append(f"{e.title} {e.company} {e.dates} {e.location} {e.summary_line}")
            parts.extend(e.bullets)
            for g in e.groups:
                parts.append(g.label)
                parts.extend(g.bullets)
        for p in self.projects:
            parts.append(f"{p.name} {p.tagline} {p.stack}")
            parts.extend(p.bullets)
        for ed in self.education:
            parts.append(f"{ed.degree} {ed.institution} {ed.dates} {ed.coursework} {ed.dissertation}")
        for c in self.certifications:
            parts.append(f"{c.name} {c.issuer} {c.year}")
        for a in self.achievements:
            parts.append(f"{a.title} {a.description} {a.year_or_org}")
        for pb in self.publications:
            parts.append(f"{pb.title} {pb.venue} {pb.year}")
        for x in self.extras:
            parts.append(x.title)
            parts.append(x.body)
            parts.extend(x.items)
        return "\n".join(p for p in parts if p)


# ----------------------------------------------------------------------
# Trace records (per-section critique loop, role review, JD match)
# ----------------------------------------------------------------------
class IterationStep(BaseModel):
    iteration: int
    draft: str
    score: float = 0.0
    dim_scores: Dict[str, float] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)
    verdict: Literal["accept", "iterate", "error"] = "iterate"
    accepted: bool = False


class SectionTrace(BaseModel):
    section_id: str         # e.g. "summary", "exp_0_bullet_2"
    label: str              # human-readable
    before: str
    after: str
    changed: bool
    final_score: float = 0.0
    iterations_used: int = 0
    iterations: List[IterationStep] = Field(default_factory=list)
    note: str = ""


class RoleReview(BaseModel):
    role_id: str
    role_name: str
    overall_score: float = 0.0          # 0-100 hiring-manager rating
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    one_line_verdict: str = ""


class JDMatch(BaseModel):
    role_id: str
    title: str
    company_archetype: str = ""
    seniority: str = ""
    keywords_total: int = 0
    keywords_matched_before: int = 0
    keywords_matched_after: int = 0
    score_before: float = 0.0
    score_after: float = 0.0
    delta: float = 0.0
    missing_keywords: List[str] = Field(default_factory=list)


class JDMatchReport(BaseModel):
    role_id: str
    samples_count: int = 0
    avg_score_before: float = 0.0
    avg_score_after: float = 0.0
    avg_delta: float = 0.0
    top_gaps: List[str] = Field(default_factory=list)
    samples: List[JDMatch] = Field(default_factory=list)


class ATSReport(BaseModel):
    score: float = 0.0
    matched_count: int = 0
    total_checked: int = 0
    matched: List[str] = Field(default_factory=list)
    missing_high_impact: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Final pipeline result
# ----------------------------------------------------------------------
class PipelineResult(BaseModel):
    job_id: str
    status: Literal["complete", "partial", "error"]
    role: str
    backend: str
    elapsed_ms: int

    ir_before: Optional[ResumeIR] = None
    ir_after: Optional[ResumeIR] = None

    tex_path: Optional[str] = None
    tex_content: str = ""

    section_traces: List[SectionTrace] = Field(default_factory=list)
    ats: Optional[ATSReport] = None
    role_reviews: List[RoleReview] = Field(default_factory=list)
    jd_report: Optional[JDMatchReport] = None
    cross_role_jd_reports: List[JDMatchReport] = Field(default_factory=list)

    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "role": self.role,
            "backend": self.backend,
            "elapsed_ms": self.elapsed_ms,
            "sections_changed": sum(1 for t in self.section_traces if t.changed),
            "sections_total": len(self.section_traces),
            "ats_score": self.ats.score if self.ats else 0.0,
            "warnings": self.warnings,
            "errors": self.errors,
        }
