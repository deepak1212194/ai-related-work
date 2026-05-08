"""
orchestrator.py - bounded Enhancer<->Critic iteration for one section.

Hard bounds (all configurable):
  - max_iterations: stop after N drafts (default 3, hard cap 4).
  - accept_threshold: accept early when score >= threshold (default 82).
  - min_delta_to_continue: stop when iteration improvement < delta.
  - time_budget_seconds: per-section deadline.

After the loop, the deterministic safety guard runs on the winning draft.
If the guard rejects, we fall back to the original text and record the
reason in `note`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.config import settings
from ..core.ir import IterationStep, SectionTrace
from ..core.safety import safe_apply
from .critic import CriticAgent
from .enhancer import EnhancerAgent

log = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    final_text: str
    accepted_text: str
    iterations: List[IterationStep] = field(default_factory=list)
    note: str = ""
    final_score: float = 0.0


class IterativeOrchestrator:
    def __init__(
        self,
        enhancer: EnhancerAgent,
        critic: Optional[CriticAgent],
        *,
        max_iterations: Optional[int] = None,
        accept_threshold: Optional[int] = None,
        min_delta: Optional[int] = None,
    ) -> None:
        self.enhancer = enhancer
        self.critic = critic
        self.max_iterations = min(
            max_iterations or settings.max_iterations, 4,
        )
        self.accept_threshold = accept_threshold or settings.accept_threshold
        self.min_delta = min_delta or settings.min_delta_to_continue

    def run(
        self,
        *,
        section_id: str,
        label: str,
        section_type: str,
        original: str,
        role_id: str,
        priority_keywords: Optional[list[str]] = None,
        is_lead_bullet: bool = False,
        time_budget_s: Optional[float] = None,
    ) -> SectionTrace:
        deadline = time.monotonic() + max(0.0, time_budget_s or settings.section_budget_s)
        steps: List[IterationStep] = []
        best_text = ""
        best_score = -1.0
        prior_critique: Optional[dict] = None
        prior_draft: Optional[str] = None

        for i in range(1, self.max_iterations + 1):
            if time.monotonic() >= deadline:
                break
            draft = self.enhancer.draft(
                section_type=section_type,
                original=original,
                role_id=role_id,
                priority_keywords=priority_keywords,
                prior_draft=prior_draft,
                prior_critique=prior_critique,
                is_lead_bullet=is_lead_bullet,
            )
            if not draft:
                steps.append(IterationStep(
                    iteration=i, draft="", score=0.0,
                    violations=["enhancer returned empty"],
                    verdict="error",
                ))
                break
            if self.critic is None or time.monotonic() >= deadline:
                steps.append(IterationStep(
                    iteration=i, draft=draft, score=80.0,
                    verdict="accept", accepted=True,
                ))
                best_text, best_score = draft, 80.0
                break
            critique = self.critic.score(section_type, original, draft)
            score = float(critique.get("total", 0))
            verdict = critique.get("verdict", "iterate")
            dim = {k: float(v) for k, v in critique.get("scores", {}).items()}
            step = IterationStep(
                iteration=i, draft=draft, score=score, dim_scores=dim,
                violations=critique.get("violations", []),
                verdict=verdict,
            )
            steps.append(step)
            if score > best_score:
                best_score = score
                best_text = draft
            if verdict == "accept" or score >= self.accept_threshold:
                step.accepted = True
                step.verdict = "accept"
                break
            if i >= 2:
                prev = steps[-2].score
                if score - prev < self.min_delta:
                    break
            prior_critique = critique
            prior_draft = draft

        # Mark winning step if no early-accept happened
        if not any(s.accepted for s in steps):
            for s in steps:
                if s.draft == best_text and s.verdict != "error":
                    s.accepted = True
                    break

        # Safety guard - drop the rewrite if it weakens or drops protected terms
        applied, guard = safe_apply(original, best_text)
        note = ""
        if not guard.ok:
            note = f"safety guard rejected: {guard.reason}"
            log.info("[Orchestrator:%s] guard rejected, kept original (%s)",
                     section_id, guard.reason)

        return SectionTrace(
            section_id=section_id,
            label=label,
            before=original,
            after=applied,
            changed=(applied != original),
            final_score=best_score if best_score >= 0 else 0.0,
            iterations_used=len(steps),
            iterations=steps,
            note=note,
        )
