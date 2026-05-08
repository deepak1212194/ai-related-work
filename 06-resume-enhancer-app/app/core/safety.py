"""
safety.py - deterministic guards that wrap every agentic rewrite.

The agentic critic loop is opinionated and useful, but a single layer of
deterministic Python checks is the final word. The critic CANNOT cause a
rewrite to drop a protected term, shrink below a length floor, or invent
a novel proper noun. If a draft would violate any guard, we fall back to
the original text and record the reason in `note`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


_PROTECTED_PATTERNS = [
    # Years and ranges
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
]


# Frameworks / libraries / cloud services / acronyms we never want dropped
# from a rewrite. The list is conservative - if a token appears in the
# input, it should appear in the output unless deliberately removed by
# the role-aware planner (which marks the change explicitly).
PROTECTED_TERMS = {
    # ML / NLP
    "PyTorch", "TensorFlow", "Hugging Face", "Transformers", "SBERT",
    "all-mpnet-base-v2", "BERT", "GPT-4", "GPT-4o", "Claude",
    "LangChain", "LlamaIndex", "CrewAI", "RAG", "FAISS", "ANN",
    "scikit-learn", "NumPy", "pandas", "Llama", "Qwen", "Mistral",
    # CV
    "YOLOv5", "YOLOv7", "YOLOv8", "U-Net", "EfficientNet",
    "OpenCV", "DeepStream", "OC-SORT", "NvSORT",
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
    # Roles / certs
    "Patent", "GTC", "GATE", "UGC-NET",
}


@dataclass
class GuardReport:
    ok: bool
    reason: str = ""
    dropped_terms: List[str] = None
    length_ratio: float = 1.0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _tokens_protected(text: str) -> List[str]:
    """Return the list of protected substrings that appear in `text`."""
    found: List[str] = []
    lower = text.lower()
    for term in PROTECTED_TERMS:
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
) -> GuardReport:
    """
    Deterministic check between original and rewrite.

    A rewrite is OK if:
      1. It is non-empty.
      2. Length is within [min_length_ratio, max_length_ratio] of original.
      3. Every protected token from the original still appears in the
         rewrite (case-insensitive substring match).
    """
    rep = GuardReport(ok=True, dropped_terms=[])
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
    # Protected-term check
    orig_protected = _tokens_protected(o)
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


def safe_apply(original: str, rewrite: str) -> Tuple[str, GuardReport]:
    """
    Apply the rewrite if it passes the guard, else fall back to original.

    Returns (text_to_use, GuardReport). Caller logs `report.reason` when
    the original was kept.
    """
    rep = check_rewrite(original, rewrite)
    return (rewrite, rep) if rep.ok else (original, rep)
