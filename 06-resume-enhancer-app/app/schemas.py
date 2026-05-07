"""
schemas.py — API contracts
============================
"""

from typing import Literal

from pydantic import BaseModel, Field


class EnhanceResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "error"]
    sections_enhanced: int
    elapsed_ms: int
    backend: str                    # "huggingface" or "anthropic"
    tex_path: str | None = None     # download URL once complete
    pdf_path: str | None = None
    pdf_compiled: bool = False      # PDF compilation may fail even if tex is fine
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    backend: str
    backend_configured: bool
    pdflatex_available: bool


class ParsedResume(BaseModel):
    """Internal — what the parser extracts from PDF or .tex input."""

    name: str = ""
    contact_line: str = ""
    summary: str = ""
    skills: dict[str, str] = Field(default_factory=dict)   # bucket → comma-separated
    experience_blocks: list["ExperienceBlock"] = Field(default_factory=list)
    education_blocks: list["EducationBlock"] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


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
    note: str = ""              # dissertation, coursework
