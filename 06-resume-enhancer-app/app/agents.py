"""
agents.py — Enhancer + Critic agents with bounded iteration
==============================================================

The resume enhancer uses a **2-agent hybrid pattern**:

    EnhancerAgent  ──draft──▶  CriticAgent  ──score──▶  Orchestrator
         ▲                                                  │
         └──────────── critique fed back ───────────────────┘

The CriticAgent encodes the rubric we wrote into `core_rules.md`
and `section_tasks.md` (action-verb strength, specificity, honesty,
tightness, ATS keyword retention) and returns a structured JSON
score. The IterativeOrchestrator runs the loop with hard bounds:

    * max_iterations         (default 3)
    * accept_threshold       (default 80/100 — accept early)
    * min_delta_to_continue  (default 3 — early-stop on diminishing returns)

Why two agents and not one shot?
--------------------------------
Subjective dimensions (verb-tier, scope phrase, "differentiation")
are exactly what an LLM is good at *judging*. A draft+critique loop
catches misses that a single shot can't (e.g., draft used "Worked
on" — critic flags it — iteration 2 leads with "Architected").

Why not full multi-agent (CrewAI)?
----------------------------------
Overkill. Resume rewriting is a focused task. Two agents with
deterministic Python guards (length ratio, protected-term drop)
gives us all the lift of agentic critique without the cost,
latency, and indeterminism of a 4-agent crew.

The deterministic guards in `enhancer._safe_replace` remain the
final gate — even if the critic accepts a draft that drops a
protected term, the safe-replace check will reject it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import settings
from .llm import LLMClient, LLMError
from .schemas import IterationStep

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Critic prompt — section-agnostic, scores 5 dimensions × 20 = 100
# ──────────────────────────────────────────────────────────────────────
CRITIC_SYSTEM_PROMPT = """You are a senior resume reviewer grading one rewrite.
You output ONLY a JSON object. No prose, no markdown fences.

You score the REWRITE against the ORIGINAL on 5 dimensions, 0-20 each:

  1. honesty (0-20):
     - 20 if every fact in REWRITE traces to ORIGINAL
     - subtract 10 for any fabricated metric/number/date
     - subtract 5 for any invented company/product/institution name
     - subtract 5 for promoting team work to solo work

  2. action_verb (0-20):
     - 20 for senior past-tense lead (Architected/Engineered/Designed/
       Owned/Co-invented/Productionized/Shipped/Migrated/Built)
     - 10 if generic but acceptable (Developed/Created/Implemented)
     - 0 for "Worked on", "Helped with", "Responsible for",
       "Built and shipped" (redundant doublet)
     - N/A for non-bullet sections — return 20

  3. specificity (0-20):
     - 20 if a specific system/scope/stack/technique is named
     - 10 if mostly vague but at least one keyword present
     - 0 for "improved performance", "managed cloud infra",
       "drove results", "various tools"

  4. tightness (0-20):
     - 20 if no filler, no hedging ("kind of", "various", "etc.")
     - subtract 5 for each filler phrase
     - subtract 10 if length > 360 chars for a bullet

  5. keyword_retention (0-20):
     - 20 if every tech term/number/proper-noun from ORIGINAL appears
       in REWRITE (case-insensitive)
     - subtract 5 per dropped term up to 20

Total = sum (0-100). Verdict = "accept" if total >= 80 OR if
violations is empty; else "iterate".

Output schema (NO prose, NO markdown, JSON ONLY):
{
  "scores": {
    "honesty": int,
    "action_verb": int,
    "specificity": int,
    "tightness": int,
    "keyword_retention": int
  },
  "total": int,
  "violations": [string, ...],
  "fix_hint": string,
  "verdict": "accept" | "iterate"
}
"""


# ──────────────────────────────────────────────────────────────────────
# Enhancer
# ──────────────────────────────────────────────────────────────────────
class EnhancerAgent:
    """
    Drafts the rewrite. Reuses the existing skill-driven system prompt
    (core_rules + role profile) and per-section task template.

    On iterations > 1, prior critique is appended to the user message
    so the model can address specific violations.
    """

    def __init__(self, llm: LLMClient, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def draft(
        self,
        task_prompt: str,
        prior_draft: Optional[str] = None,
        prior_critique: Optional[dict] = None,
    ) -> str:
        user = task_prompt
        if prior_draft is not None and prior_critique is not None:
            violations = prior_critique.get("violations", [])
            fix_hint = prior_critique.get("fix_hint", "")
            critique_block = (
                "\n\n---\n\nPREVIOUS DRAFT (rejected):\n"
                f'"""{prior_draft}"""\n\n'
                "CRITIC FEEDBACK:\n"
            )
            if violations:
                critique_block += "Violations:\n"
                for v in violations:
                    critique_block += f"  - {v}\n"
            if fix_hint:
                critique_block += f"\nFix hint: {fix_hint}\n"
            critique_block += (
                "\nProduce a NEW draft that addresses every violation. "
                "Keep all protected terms and numbers from the original. "
                "Plain text only — no quotes, no preamble.\n"
            )
            user = task_prompt + critique_block
        try:
            return self.llm.complete(self.system_prompt, user).strip()
        except LLMError as e:
            log.warning("[Enhancer] draft failed: %s", e)
            return ""
        except Exception as e:                              # noqa: BLE001
            log.warning("[Enhancer] unexpected error: %s", e)
            return ""


# ──────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────
class CriticAgent:
    """
    Scores a (before, after) pair. Returns dict with:
        scores: dict[str, int]
        total: int
        violations: list[str]
        fix_hint: str
        verdict: "accept" | "iterate"

    On any failure (LLM error, JSON parse error), returns a
    permissive accept with verdict="accept" and total=80, so the
    pipeline degrades to single-call behavior rather than blocking.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _build_user_msg(section_type: str, before: str, after: str) -> str:
        return (
            f'SECTION TYPE: {section_type}\n\n'
            f'ORIGINAL:\n"""\n{before}\n"""\n\n'
            f'REWRITE:\n"""\n{after}\n"""\n\n'
            "Output the JSON now."
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Robust JSON extraction — strips markdown fences and stray prose."""
        if not text:
            return None
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Strip markdown fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Find first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def score(self, section_type: str, before: str, after: str) -> dict:
        if not after.strip():
            return self._fallback_accept(reason="empty draft")

        user = self._build_user_msg(section_type, before, after)
        try:
            raw = self.llm.complete(CRITIC_SYSTEM_PROMPT, user)
        except LLMError as e:
            log.warning("[Critic] LLM error: %s", e)
            return self._fallback_accept(reason="llm_error")
        except Exception as e:                              # noqa: BLE001
            log.warning("[Critic] unexpected error: %s", e)
            return self._fallback_accept(reason="exception")

        parsed = self._extract_json(raw)
        if parsed is None:
            log.debug("[Critic] could not parse JSON; raw=%s", raw[:200])
            return self._fallback_accept(reason="parse_error")

        # Validate shape
        scores = parsed.get("scores", {}) or {}
        try:
            total = int(parsed.get("total") or sum(int(v) for v in scores.values()))
        except Exception:
            total = 0
        violations = parsed.get("violations") or []
        if not isinstance(violations, list):
            violations = [str(violations)]
        verdict = parsed.get("verdict") or ("accept" if total >= 80 else "iterate")
        if verdict not in ("accept", "iterate"):
            verdict = "iterate"
        fix_hint = parsed.get("fix_hint") or ""

        return {
            "scores": {k: int(v) for k, v in scores.items() if isinstance(v, (int, float))},
            "total": max(0, min(100, total)),
            "violations": [str(v)[:200] for v in violations[:6]],
            "fix_hint": str(fix_hint)[:300],
            "verdict": verdict,
        }

    @staticmethod
    def _fallback_accept(reason: str) -> dict:
        # Permissive default so the pipeline keeps moving
        return {
            "scores": {},
            "total": 80,
            "violations": [],
            "fix_hint": f"(critic unavailable: {reason})",
            "verdict": "accept",
        }


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OrchestratorResult:
    """Result of one orchestrated section enhancement."""

    best_text: str                              # winning draft (or original if all rejected)
    iterations: list[IterationStep] = field(default_factory=list)
    used_iterations: int = 0


class IterativeOrchestrator:
    """
    Runs the Enhancer→Critic loop for one section.

    Hard bounds (all configurable via settings):
      * max_iterations          — 3 by default
      * accept_threshold        — 80/100; first draft scoring at or above stops
      * min_delta_to_continue   — early-stop when score improvement < this
      * time_budget_seconds     — caller passes per-section budget; loop exits
                                  when budget is exhausted

    The orchestrator does NOT apply length/keyword guards. Those run
    in `enhancer._safe_replace` after the orchestrator returns. This
    separation keeps deterministic guarantees independent of what the
    critic decides.
    """

    def __init__(self, enhancer: EnhancerAgent, critic: Optional[CriticAgent]) -> None:
        self.enhancer = enhancer
        self.critic = critic
        self.max_iterations = settings.agent_max_iterations
        self.accept_threshold = settings.agent_accept_threshold
        self.min_delta = settings.agent_min_delta_to_continue

    def run(
        self,
        section_type: str,
        original: str,
        task_prompt: str,
        time_budget_seconds: float,
    ) -> OrchestratorResult:
        """
        section_type:   "summary" | "bullet" | "skills" | "achievement"
        original:       the input text we're rewriting
        task_prompt:    the user message (already contains the input)
        time_budget_seconds: per-section budget; loop exits when exhausted
        """
        deadline = time.monotonic() + max(0.0, time_budget_seconds)
        steps: list[IterationStep] = []
        best_text = ""
        best_score = -1.0
        prior_critique: Optional[dict] = None
        prior_draft: Optional[str] = None

        for i in range(1, self.max_iterations + 1):
            if time.monotonic() >= deadline:
                break

            draft = self.enhancer.draft(
                task_prompt=task_prompt,
                prior_draft=prior_draft,
                prior_critique=prior_critique,
            )
            if not draft:
                steps.append(IterationStep(
                    iteration=i, draft="", score=0.0, dim_scores={},
                    violations=["enhancer returned empty"], verdict="error",
                    accepted=False,
                ))
                break

            # If critic disabled or out of budget, accept this draft
            if self.critic is None or time.monotonic() >= deadline:
                steps.append(IterationStep(
                    iteration=i, draft=draft, score=80.0, dim_scores={},
                    violations=[], verdict="accept", accepted=True,
                ))
                best_text, best_score = draft, 80.0
                break

            critique = self.critic.score(section_type, original, draft)
            score = float(critique.get("total", 0))
            verdict = critique.get("verdict", "accept")
            dim_scores = {k: float(v) for k, v in critique.get("scores", {}).items()}
            violations = critique.get("violations", [])

            step = IterationStep(
                iteration=i, draft=draft, score=score, dim_scores=dim_scores,
                violations=violations, verdict=verdict, accepted=False,
            )
            steps.append(step)

            # Track best
            if score > best_score:
                best_score = score
                best_text = draft

            # Accept early
            if verdict == "accept" or score >= self.accept_threshold:
                step.accepted = True
                step.verdict = "accept"
                break

            # Early-stop on diminishing returns (after iteration 1)
            if i >= 2:
                prev = steps[-2].score
                if score - prev < self.min_delta:
                    break

            prior_critique = critique
            prior_draft = draft

        # Mark winning step
        if not any(s.accepted for s in steps):
            for s in steps:
                if s.draft == best_text and s.verdict != "error":
                    s.accepted = True
                    break

        return OrchestratorResult(
            best_text=best_text,
            iterations=steps,
            used_iterations=len(steps),
        )
