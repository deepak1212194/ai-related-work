"""
safety.py - deterministic guards that wrap every agentic rewrite.

The agentic critic loop is opinionated and useful, but a single layer of
deterministic Python checks is the final word. The critic CANNOT cause a
rewrite to drop a protected term, shrink below a length floor, or invent
a novel proper noun. If a draft would violate any guard, we fall back to
the original text and record the reason in `note`.

Protected terms are now DYNAMIC — extracted from the user's own parsed
resume IR — so every user's frameworks, tools, and proper nouns are
guarded, not just a hardcoded list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Set, Tuple

if TYPE_CHECKING:
    from .ir import ResumeIR


_BLACKLISTED_PHRASES: List[str] = [
    "spearheaded", "leveraged", "harnessed", "cutting-edge", "revolutionized",
    "transformed", "synergized", "utilized", "innovative", "passionate", "driven",
    "proven track record", "strong background", "excellent communication",
    "team player", "fast-paced", "results-driven", "detail-oriented",
    "extensive experience", "unique ability", "best-in-class", "world-class",
    "state-of-the-art", "dynamic", "motivated", "forward-thinking", "game-changing",
]

_BLACKLISTED_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _BLACKLISTED_PHRASES) + r")\b",
    re.IGNORECASE,
)

_PROTECTED_PATTERNS = [
    # Calendar years and year ranges
    re.compile(r"\b(19|20)\d{2}\b"),
    # Percentages and X-fold metrics
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?x\b", re.IGNORECASE),
    # Counts with units (M, K, B, GB, MB, ms)
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:M|K|B|MM|GB|MB|KB|TB|ms|s|GPUs?|nodes?)\b"),
    # Dollar / currency amounts
    re.compile(r"[$€£]\s?\d[\d,.]*"),
    # Patent / claim counts
    re.compile(r"\b\d+\s+claims?\b", re.IGNORECASE),
    # Years / months of experience — "3 years", "5+ years", "18 months"
    re.compile(r"\b\d+\+?\s*(?:years?|months?|weeks?)\b", re.IGNORECASE),
    # Ordinal rankings — "top 1%", "rank 2"
    re.compile(r"\b(?:top|rank)\s+\d+", re.IGNORECASE),
    # Team / headcount numbers — "team of 8", "across 12 teams"
    re.compile(r"\b(?:team|squad|group|org|department)\s+of\s+\d+", re.IGNORECASE),
    # Version numbers — v2.3, Python 3.10
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
]


# Baseline protected terms — these are always checked regardless of the
# user's resume content. They cover extremely common ML/cloud/dev tools.
_BASELINE_PROTECTED_TERMS: Set[str] = {
    # ML / NLP
    "PyTorch", "TensorFlow", "Hugging Face", "Transformers", "SBERT",
    "BERT", "GPT-4", "GPT-4o", "Claude", "LangChain", "LlamaIndex",
    "CrewAI", "RAG", "FAISS", "ANN", "scikit-learn", "NumPy", "pandas",
    "Llama", "Qwen", "Mistral",
    # CV
    "YOLOv5", "YOLOv7", "YOLOv8", "U-Net", "EfficientNet",
    "OpenCV", "DeepStream",
    # Cloud
    "Azure", "AKS", "Azure ML", "Azure OpenAI", "Azure AI Search",
    "Event Hub", "DevOps", "AWS", "GCP", "S3", "EKS", "GKE",
    "Lambda", "Kubernetes", "Docker", "Terraform",
    # GPU / edge
    "NVIDIA", "DGX", "Jetson", "Triton", "TensorRT", "ONNX",
    "CUDA", "FP8", "FP16", "INT8", "NIM",
    # Languages / DBs
    "Python", "C++", "SQL", "PostgreSQL", "MySQL", "Cosmos DB",
    "MongoDB", "Redis", "Kafka",
    # Web
    "FastAPI", "Flask", "Django", "React", "Angular", "Next.js",
    # Additional common tools
    "Spring Boot", "Vue.js", "Svelte", "Rust", "Go", "Java",
    "Swift", "Kotlin", "TypeScript", "JavaScript", "Node.js",
    "GraphQL", "REST", "gRPC", "Spark", "Airflow", "MLflow",
    "Weights & Biases", "W&B", "DVC", "Prometheus", "Grafana",
    "Elasticsearch", "Solr", "Pinecone", "Weaviate", "Milvus",
    "Snowflake", "BigQuery", "Databricks", "Tableau", "Power BI",
    "Unity", "Unreal", "Solidity", "Hardhat",
}

# Minimum length for a term to be considered for dynamic extraction
_MIN_TERM_LENGTH = 2


def extract_protected_terms_from_ir(ir: "ResumeIR") -> Set[str]:
    """Extract protected terms dynamically from the user's parsed resume.

    This ensures that EVERY user's specific tools, frameworks, company
    names, and proper nouns are protected — not just a hardcoded list.
    """
    terms: Set[str] = set()

    # 1. Skills — every item in every bucket
    for bucket in ir.skills:
        terms.add(bucket.name)
        for item in bucket.items:
            item = item.strip()
            if len(item) >= _MIN_TERM_LENGTH:
                terms.add(item)

    # 2. Experience — company names, titles, group labels
    for exp in ir.experience:
        if exp.company:
            terms.add(exp.company.strip())
        if exp.title:
            terms.add(exp.title.strip())
        for grp in exp.groups:
            if grp.label:
                terms.add(grp.label.strip())

    # 3. Projects — names, stack items
    for proj in ir.projects:
        if proj.name:
            terms.add(proj.name.strip())
        if proj.stack:
            # Stack is often comma-separated
            for item in proj.stack.split(","):
                item = item.strip()
                if len(item) >= _MIN_TERM_LENGTH:
                    terms.add(item)

    # 4. Education — institution names, degree names
    for ed in ir.education:
        if ed.institution:
            terms.add(ed.institution.strip())
        if ed.degree:
            terms.add(ed.degree.strip())

    # 5. Certifications — names, issuers
    for cert in ir.certifications:
        if cert.name:
            terms.add(cert.name.strip())
        if cert.issuer:
            terms.add(cert.issuer.strip())

    # 6. Achievements — titles
    for ach in ir.achievements:
        if ach.title:
            terms.add(ach.title.strip())

    # 7. Publications — titles, venues
    for pub in ir.publications:
        if pub.title:
            terms.add(pub.title.strip())
        if pub.venue:
            terms.add(pub.venue.strip())

    # 8. Header — name, proper nouns
    if ir.header.name:
        terms.add(ir.header.name.strip())

    # Filter out very short or empty terms and generic words
    _GENERIC = {"and", "the", "for", "with", "from", "into", "using", "via", "at", "in", "on", "to", "of", "a", "an"}
    filtered = {
        t for t in terms
        if len(t) >= _MIN_TERM_LENGTH
        and t.lower() not in _GENERIC
        and not t.startswith("[")  # Skip placeholders
    }

    return filtered


def get_all_protected_terms(ir: "ResumeIR | None" = None) -> Set[str]:
    """Return the union of baseline + dynamic protected terms."""
    terms = set(_BASELINE_PROTECTED_TERMS)
    if ir is not None:
        terms |= extract_protected_terms_from_ir(ir)
    return terms


@dataclass
class GuardReport:
    ok: bool
    reason: str = ""
    dropped_terms: List[str] = field(default_factory=list)
    length_ratio: float = 1.0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _tokens_protected(text: str, protected_terms: Set[str] | None = None) -> List[str]:
    """Return the list of protected substrings that appear in `text`."""
    terms = protected_terms or _BASELINE_PROTECTED_TERMS
    found: List[str] = []
    lower = text.lower()
    for term in terms:
        if term.lower() in lower:
            found.append(term)
    for pat in _PROTECTED_PATTERNS:
        for m in pat.finditer(text):
            found.append(m.group(0))
    return found


def check_rewrite(
    original: str,
    rewrite: str,
    *,
    min_length_ratio: float = 0.5,
    max_length_ratio: float = 4.0,
    protected_terms: Set[str] | None = None,
) -> GuardReport:
    """
    Deterministic check between original and rewrite.

    A rewrite is OK if:
      1. It is non-empty.
      2. Length is within [min_length_ratio, max_length_ratio] of original.
      3. No blacklisted buzzwords were introduced.
      4. Every protected token that appeared in the ORIGINAL still appears in
         the rewrite. Terms from protected_terms that were NOT in the original
         are ignored — the enhancer is allowed to add new skills from USER_SKILLS.
    """
    rep = GuardReport(ok=True)
    o, r = _normalize(original), _normalize(rewrite)
    if not r:
        rep.ok = False
        rep.reason = "rewrite is empty"
        return rep
    if len(o) > 0:
        ratio = len(r) / max(1, len(o))
        rep.length_ratio = ratio
        if ratio < min_length_ratio:
            rep.ok = False
            rep.reason = f"rewrite shrank to {ratio:.0%} of original (< {min_length_ratio:.0%})"
            return rep
        if ratio > max_length_ratio:
            rep.ok = False
            rep.reason = f"rewrite grew to {ratio:.1f}x original (> {max_length_ratio:.1f}x)"
            return rep
    # Blacklisted-phrase check — reject rewrites that introduce buzzwords
    bm = _BLACKLISTED_RE.search(r)
    if bm:
        rep.ok = False
        rep.reason = f"rewrite contains blacklisted phrase: '{bm.group(0)}'"
        return rep
    # Protected-term check — only terms that were present in the ORIGINAL
    # bullet are checked. Terms from the global protected set that were absent
    # from the original are intentionally excluded: the enhancer may legitimately
    # add skills from USER_SKILLS into a bullet where they weren't mentioned.
    orig_protected = _tokens_protected(o, protected_terms)
    new_lower = r.lower()
    dropped: List[str] = []
    for tok in orig_protected:
        if tok.lower() not in new_lower:
            dropped.append(tok)
    if dropped:
        rep.ok = False
        rep.dropped_terms = dropped
        rep.reason = f"rewrite dropped protected term(s): {', '.join(dropped[:5])}"
        return rep
    return rep


def safe_apply(
    original: str,
    rewrite: str,
    *,
    protected_terms: Set[str] | None = None,
) -> Tuple[str, GuardReport]:
    """
    Apply the rewrite if it passes the guard, else fall back to original.

    Returns (text_to_use, GuardReport). Caller logs `report.reason` when
    the original was kept.
    """
    rep = check_rewrite(original, rewrite, protected_terms=protected_terms)
    return (rewrite, rep) if rep.ok else (original, rep)
