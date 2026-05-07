"""
schemas.py — API contracts
============================
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
class RoleInfo(BaseModel):
    id: str
    name: str
    description: str


class IterationStep(BaseModel):
    """One pass of the Enhancer→Critic loop for a section."""

    iteration: int                                  # 1, 2, 3
    draft: str                                      # text produced this iteration
    score: float = 0.0                              # critic total score (0-100)
    dim_scores: Dict[str, float] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)
    verdict: Literal["accept", "iterate", "error"] = "accept"
    accepted: bool = False                          # this is the chosen iteration


class SectionPreview(BaseModel):
    """Before/after snapshot for one section, shown inline in the UI."""

    section: str
    before: str
    after: str
    changed: bool
    note: str = ""        # e.g. "kept (guard tripped)"
    # Agentic loop trace (empty if critic disabled or section uses single-call path)
    iterations: List[IterationStep] = Field(default_factory=list)
    final_score: float = 0.0
    iterations_used: int = 0


class ATSScore(BaseModel):
    """ATS keyword coverage analysis."""
    score: float = 0.0
    matched_count: int = 0
    total_checked: int = 0
    matched: List[str] = Field(default_factory=list)
    missing_high_impact: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class EnhanceResponse(BaseModel):
    job_id: str
    status: Literal["complete", "partial", "error"]
    role: str
    sections_enhanced: int
    sections_total: int
    elapsed_ms: int
    backend: str
    tex_path: str | None = None
    pdf_path: str | None = None
    pdf_compiled: bool = False
    notes: list[str] = Field(default_factory=list)
    previews: list[SectionPreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ats_score: Optional[ATSScore] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    backend: str
    backend_configured: bool
    pdflatex_available: bool
    available_roles: list[RoleInfo] = Field(default_factory=list)
    skill_files_loaded: int = 0


class SkillInfoResponse(BaseModel):
    loaded_files: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    section_tasks: List[str] = Field(default_factory=list)
    core_rules_length: int = 0


# ──────────────────────────────────────────────────────────────────────
# Internal — passed between parser, enhancer, compiler
# ──────────────────────────────────────────────────────────────────────
class ExperienceBlock(BaseModel):
    title: str
    company: str
    location: str = ""
    dates: str = ""
    bullets: list[str] = Field(default_factory=list)


class EducationBlock(BaseModel):
    degree: str
    institution: str
    location: str = ""
    dates: str = ""
    note: str = ""


class ParsedResume(BaseModel):
    name: str = ""
    contact_line: str = ""
    summary: str = ""
    skills: dict[str, str] = Field(default_factory=dict)
    experience_blocks: list[ExperienceBlock] = Field(default_factory=list)
    education_blocks: list[EducationBlock] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
