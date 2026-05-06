"""
orchestrator.py — Sequential Coordinator
=========================================
Multi-Agent Research Crew — Module 3

Runs the crew in order, passing a shared context dict between agents.
The orchestrator owns the LLM client and the (optional) tool routing
for the researcher agent.

Falls back to a deterministic stub when OPENAI_API_KEY is not set, so
the orchestration trace is always inspectable.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from .agents import Agent, CREW
from .tools import TOOLS, web_search_stub

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 600


# ──────────────────────────────────────────────────────────────────────
# Shared context passed between agents
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CrewContext:
    topic: str
    plan: str = ""
    research: str = ""
    critique: str = ""
    final_brief: str = ""
    trace: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Module 1: LLM call (with offline fallback)
# ──────────────────────────────────────────────────────────────────────
def _call_llm(system: str, user: str) -> str:
    """Call OpenAI when a key is present, else return a deterministic stub."""
    if not os.environ.get("OPENAI_API_KEY"):
        return _offline_response(system, user)

    from openai import OpenAI
    client = OpenAI()

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def _offline_response(system: str, user: str) -> str:
    """Tiny stub keyed off the role keyword in the system prompt."""
    s = system.lower()
    if "planner" in s:
        return (
            "1. Define the question scope.\n"
            "2. Identify two competing positions.\n"
            "3. Surface evidence supporting each.\n"
            "4. Note open questions."
        )
    if "researcher" in s:
        return (
            "1. (offline) " + web_search_stub("moe") + "\n"
            "2. (offline) " + web_search_stub("rag") + "\n"
            "3. (offline) " + web_search_stub("agent")
        )
    if "critic" in s:
        return (
            "Weaknesses: offline mode produced canned snippets; no live "
            "evidence; subtask 4 was not actually answered."
        )
    if "writer" in s:
        return (
            "## Summary\nOffline-mode brief based on stub research.\n\n"
            "- MoE LLMs reduce per-token cost but raise memory cost.\n"
            "- RAG reduces hallucination at the price of retrieval latency.\n"
            "- Multi-agent crews scale composition better than single-agent loops.\n\n"
            "## Caveats\nGenerated in offline-stub mode; all evidence is canned."
        )
    return "(offline stub)"


# ──────────────────────────────────────────────────────────────────────
# Module 2: Per-agent runner
# ──────────────────────────────────────────────────────────────────────
def _run_agent(agent: Agent, user_msg: str, ctx: CrewContext) -> str:
    """Run a single agent and append a trace entry."""
    print(f"[CREW] → {agent.name} ({agent.role})")
    output = _call_llm(agent.system, user_msg)
    ctx.trace.append(f"### {agent.name}\n{output}\n")

    # Only the researcher is allowed to invoke tools; we expose a single
    # post-hoc tool call as a demonstration. A production version would
    # use OpenAI tool-calls and a multi-turn loop.
    if agent.can_use_tools:
        snippet = TOOLS["web_search_stub"](ctx.topic)
        output += f"\n\n_(tool: web_search_stub) → {snippet}_"

    return output


# ──────────────────────────────────────────────────────────────────────
# Module 3: Public entry
# ──────────────────────────────────────────────────────────────────────
def run_crew(topic: str) -> CrewContext:
    """Run the four agents in sequence and return the populated context."""
    ctx = CrewContext(topic=topic)

    # 1. Planner
    ctx.plan = _run_agent(CREW[0], f"Topic: {topic}", ctx)
    # 2. Researcher
    ctx.research = _run_agent(
        CREW[1],
        f"Topic: {topic}\n\nSubtasks:\n{ctx.plan}",
        ctx,
    )
    # 3. Critic
    ctx.critique = _run_agent(
        CREW[2],
        f"Subtasks:\n{ctx.plan}\n\nResearch:\n{ctx.research}",
        ctx,
    )
    # 4. Writer
    ctx.final_brief = _run_agent(
        CREW[3],
        (
            f"Topic: {topic}\n\nResearch:\n{ctx.research}\n\n"
            f"Critique:\n{ctx.critique}"
        ),
        ctx,
    )
    return ctx
