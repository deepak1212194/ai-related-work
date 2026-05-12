"""
test_safety.py - unit tests for the deterministic safety guard.

Tests both the static baseline terms and the dynamic extraction from IR.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.safety import (
    check_rewrite,
    safe_apply,
    extract_protected_terms_from_ir,
    get_all_protected_terms,
)
from app.core.ir import (
    ResumeIR,
    HeaderInfo,
    SkillBucket,
    ExperienceBlock,
    ProjectBlock,
    EducationBlock,
    CertificationItem,
    AchievementItem,
    PublicationItem,
)


def _make_ir():
    """Create a sample IR for testing dynamic extraction."""
    return ResumeIR(
        header=HeaderInfo(name="John Doe"),
        skills=[
            SkillBucket(name="Languages", items=["Python", "C++", "Rust"]),
            SkillBucket(name="ML Frameworks", items=["PyTorch", "TensorFlow", "scikit-learn"]),
        ],
        experience=[
            ExperienceBlock(
                title="ML Engineer",
                company="Acme Corp",
                bullets=["Built a recommendation engine using FAISS"],
            ),
        ],
        projects=[
            ProjectBlock(name="Resume Enhancer", stack="Gradio, Jinja2, Pydantic"),
        ],
        education=[
            EducationBlock(degree="M.S. Computer Science", institution="MIT"),
        ],
        certifications=[
            CertificationItem(name="AWS Solutions Architect", issuer="Amazon"),
        ],
        achievements=[
            AchievementItem(title="Best Paper Award"),
        ],
        publications=[
            PublicationItem(title="On Neural Architectures", venue="NeurIPS"),
        ],
    )


class TestCheckRewrite:
    def test_empty_rewrite_rejected(self):
        r = check_rewrite("original text", "")
        assert not r.ok
        assert "empty" in r.reason

    def test_identical_passes(self):
        r = check_rewrite("some text", "some text")
        assert r.ok

    def test_shrink_below_50pct_rejected(self):
        r = check_rewrite("a" * 100, "a" * 40)
        assert not r.ok
        assert "shrank" in r.reason

    def test_growth_above_4x_rejected(self):
        r = check_rewrite("short", "x" * 100)
        assert not r.ok
        assert "grew" in r.reason

    def test_dropped_year_rejected(self):
        original = "Built the system in 2023 using PyTorch"
        rewrite = "Built the system using PyTorch"
        r = check_rewrite(original, rewrite)
        assert not r.ok
        assert r.dropped_terms  # Should have dropped "2023"

    def test_dropped_percentage_rejected(self):
        original = "Improved accuracy by 15%"
        rewrite = "Improved accuracy significantly"
        r = check_rewrite(original, rewrite)
        assert not r.ok

    def test_preserves_all_terms_passes(self):
        original = "Built with PyTorch and FAISS in 2023, achieving 95% accuracy"
        rewrite = "Engineered a system with PyTorch and FAISS in 2023, achieving 95% accuracy for real-time search"
        r = check_rewrite(original, rewrite)
        assert r.ok

    def test_custom_protected_terms(self):
        terms = {"Spring Boot", "Vue.js"}
        original = "Built web app with Spring Boot and Vue.js"
        rewrite = "Built web app with Django and React"
        r = check_rewrite(original, rewrite, protected_terms=terms)
        assert not r.ok
        assert any("Spring Boot" in t for t in r.dropped_terms)


class TestSafeApply:
    def test_good_rewrite_applied(self):
        text, report = safe_apply("original text here", "better text here version")
        assert text == "better text here version"
        assert report.ok

    def test_bad_rewrite_falls_back(self):
        text, report = safe_apply("original text here", "")
        assert text == "original text here"
        assert not report.ok


class TestDynamicExtraction:
    def test_extracts_skills(self):
        ir = _make_ir()
        terms = extract_protected_terms_from_ir(ir)
        assert "Python" in terms
        assert "PyTorch" in terms
        assert "Rust" in terms

    def test_extracts_company(self):
        ir = _make_ir()
        terms = extract_protected_terms_from_ir(ir)
        assert "Acme Corp" in terms

    def test_extracts_project_stack(self):
        ir = _make_ir()
        terms = extract_protected_terms_from_ir(ir)
        assert "Gradio" in terms
        assert "Jinja2" in terms

    def test_extracts_education(self):
        ir = _make_ir()
        terms = extract_protected_terms_from_ir(ir)
        assert "MIT" in terms

    def test_extracts_certification(self):
        ir = _make_ir()
        terms = extract_protected_terms_from_ir(ir)
        assert "AWS Solutions Architect" in terms

    def test_filters_generic_words(self):
        ir = _make_ir()
        terms = extract_protected_terms_from_ir(ir)
        assert "and" not in terms
        assert "the" not in terms

    def test_get_all_includes_baseline(self):
        ir = _make_ir()
        all_terms = get_all_protected_terms(ir)
        # Should have baseline + dynamic
        assert "Docker" in all_terms  # baseline
        assert "Acme Corp" in all_terms  # dynamic


class TestCoerceScore:
    """Test the coerce_score function from base.py."""

    def test_integer_1_stays_1(self):
        from app.agents.base import coerce_score
        assert coerce_score(1) == 1.0

    def test_integer_82_stays_82(self):
        from app.agents.base import coerce_score
        assert coerce_score(82) == 82.0

    def test_float_fraction_expanded(self):
        from app.agents.base import coerce_score
        assert coerce_score(0.85) == 85.0

    def test_string_82_stays_82(self):
        from app.agents.base import coerce_score
        assert coerce_score("82") == 82.0

    def test_string_fraction_82_100(self):
        from app.agents.base import coerce_score
        assert coerce_score("82/100") == 82.0

    def test_string_fraction_8_10(self):
        from app.agents.base import coerce_score
        assert coerce_score("8/10") == 80.0

    def test_none_returns_0(self):
        from app.agents.base import coerce_score
        assert coerce_score(None) == 0.0

    def test_capped_at_scale(self):
        from app.agents.base import coerce_score
        assert coerce_score(150) == 100.0
