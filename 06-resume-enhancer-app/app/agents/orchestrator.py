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

Safety guard now uses DYNAMIC protected terms extracted from the user's
resume, not just a hardcoded list.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

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
        protected_terms: Optional[Set[str]] = None,
    ) -> None:
        self.enhancer = enhancer
        self.critic = critic
        self.max_iterations = min(
            max_iterations or settings.max_iterations, 4,   # hard cap: 4 iterations max
        )
        self.accept_threshold = accept_threshold or settings.accept_threshold
        self.min_delta = min_delta or settings.min_delta_to_continue
        self.protected_terms = protected_terms

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
        skills_context: Optional[str] = None,
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
            is_final = (i == self.max_iterations)
            draft = self.enhancer.draft(
                section_type=section_type,
                original=original,
                role_id=role_id,
                priority_keywords=priority_keywords,
                prior_draft=prior_draft,
                prior_critique=prior_critique,
                is_lead_bullet=is_lead_bullet,
                skills_context=skills_context,
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
            critique = self.critic.score(
                section_type, original, draft,
                is_final_iteration=is_final,
            )
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
        applied, guard = safe_apply(
            original, best_text,
            protected_terms=self.protected_terms,
        )
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

    # ------------------------------------------------------------------
    # Block-level batching: one LLM call per block instead of per bullet
    # ------------------------------------------------------------------

    def run_block(
        self,
        *,
        block_id: str,
        block_context: str,
        section_type: str,
        originals: list[str],
        labels: list[str],
        section_ids: list[str],
        role_id: str,
        priority_keywords: Optional[list[str]] = None,
        lead_indices: Optional[Set[int]] = None,
        time_budget_s: Optional[float] = None,
        skills_context: Optional[str] = None,
    ) -> List[SectionTrace]:
        """Enhance an entire experience/project block in one LLM call per iteration.

        Returns one SectionTrace per bullet — same schema as calling run()
        individually — but uses 1–4 LLM calls for the whole block instead of
        1–4 per bullet, giving a 3–5× reduction for typical blocks.

        If the block draft fails (empty return or wrong count) the method
        transparently falls back to per-bullet run() calls so accuracy is
        never compromised.
        """
        if not originals:
            return []

        n        = len(originals)
        deadline = time.monotonic() + max(
            0.0, time_budget_s or settings.section_budget_s * n
        )

        # per-bullet state across iterations
        steps_per: List[List[IterationStep]] = [[] for _ in range(n)]
        best_drafts: List[str]  = [""] * n
        best_scores: List[float] = [-1.0] * n
        prior_critique: Optional[dict]       = None
        prior_drafts:   Optional[List[str]]  = None

        for i in range(1, self.max_iterations + 1):
            if time.monotonic() >= deadline:
                break
            is_final = (i == self.max_iterations)

            drafts = self.enhancer.draft_block(
                section_type=section_type,
                block_context=block_context,
                bullets=originals,
                role_id=role_id,
                priority_keywords=priority_keywords,
                prior_drafts=prior_drafts,
                prior_critique=prior_critique,
                lead_indices=lead_indices,
                skills_context=skills_context,
            )

            # Total block failure or wrong count → fall back immediately
            if not drafts or len(drafts) != n:
                log.info(
                    "[Orchestrator.block:%s] draft_block returned %d/%d — falling back to per-bullet",
                    block_id, len(drafts) if drafts else 0, n,
                )
                return self._run_block_fallback(
                    originals=originals, labels=labels, section_ids=section_ids,
                    section_type=section_type, role_id=role_id,
                    priority_keywords=priority_keywords,
                    lead_indices=lead_indices, skills_context=skills_context,
                )

            # Fill any empty slots with the original so the critic has full context
            drafts = [d if d else originals[j] for j, d in enumerate(drafts)]

            # No critic → accept immediately
            if self.critic is None or time.monotonic() >= deadline:
                for j, draft in enumerate(drafts):
                    step = IterationStep(
                        iteration=i, draft=draft, score=80.0,
                        verdict="accept", accepted=True,
                    )
                    steps_per[j].append(step)
                    best_drafts[j] = draft
                    best_scores[j] = 80.0
                break

            critique = self.critic.score_block(
                section_type, originals, drafts,
                is_final_iteration=is_final,
            )
            score   = float(critique.get("total", 0))
            verdict = critique.get("verdict", "iterate")
            dim     = {k: float(v) for k, v in critique.get("scores", {}).items()}

            for j, draft in enumerate(drafts):
                step = IterationStep(
                    iteration=i, draft=draft, score=score, dim_scores=dim,
                    violations=critique.get("violations", []),
                    verdict=verdict,
                )
                steps_per[j].append(step)
                if score > best_scores[j]:
                    best_scores[j] = score
                    best_drafts[j] = draft

            if verdict == "accept" or score >= self.accept_threshold:
                for j in range(n):
                    if steps_per[j]:
                        steps_per[j][-1].accepted = True
                        steps_per[j][-1].verdict  = "accept"
                break

            if i >= 2:
                prev = steps_per[0][-2].score if len(steps_per[0]) >= 2 else 0.0
                if score - prev < self.min_delta:
                    break

            prior_critique = critique
            prior_drafts   = list(drafts)

        # Mark the winning step for each bullet
        for j in range(n):
            if not any(s.accepted for s in steps_per[j]):
                for s in steps_per[j]:
                    if s.draft == best_drafts[j] and s.verdict != "error":
                        s.accepted = True
                        break

        # Build per-bullet SectionTrace with individual safety guards
        traces: List[SectionTrace] = []
        for j in range(n):
            applied, guard = safe_apply(
                originals[j], best_drafts[j],
                protected_terms=self.protected_terms,
            )
            note = ""
            if not guard.ok:
                note = f"safety guard rejected: {guard.reason}"
                log.info(
                    "[Orchestrator.block:%s] bullet %d guard rejected (%s)",
                    block_id, j, guard.reason,
                )
            traces.append(SectionTrace(
                section_id=section_ids[j],
                label=labels[j],
                before=originals[j],
                after=applied,
                changed=(applied != originals[j]),
                final_score=best_scores[j] if best_scores[j] >= 0 else 0.0,
                iterations_used=len(steps_per[j]),
                iterations=steps_per[j],
                note=note,
            ))
        return traces

    def _run_block_fallback(
        self,
        *,
        originals: list[str],
        labels: list[str],
        section_ids: list[str],
        section_type: str,
        role_id: str,
        priority_keywords: Optional[list[str]] = None,
        lead_indices: Optional[Set[int]] = None,
        skills_context: Optional[str] = None,
    ) -> List[SectionTrace]:
        """Per-bullet run() calls used when block drafting fails."""
        traces: List[SectionTrace] = []
        for j, (original, label, sid) in enumerate(
            zip(originals, labels, section_ids)
        ):
            trace = self.run(
                section_id=sid,
                label=label,
                section_type=section_type,
                original=original,
                role_id=role_id,
                priority_keywords=priority_keywords,
                is_lead_bullet=(j in (lead_indices or set())),
                skills_context=skills_context,
            )
            traces.append(trace)
        return traces
