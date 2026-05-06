"""
agents.py — Agent Definitions
=============================
Multi-Agent Research Crew — Module 1

Defines the four specialist roles. Each agent is a frozen dataclass:

    name        — short identifier used in the orchestration trace
    role        — short human-readable role label
    system      — system prompt (the only thing that varies per agent)
    can_use_tools — whether the orchestrator should expose tools to it

Keeping prompts in this single file makes prompt-tuning a one-place edit.
"""

from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────
# Agent dataclass
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    system: str
    can_use_tools: bool = False


# ──────────────────────────────────────────────────────────────────────
# Module 1: Planner
# ──────────────────────────────────────────────────────────────────────
PLANNER = Agent(
    name="planner",
    role="Research Planner",
    system=(
        "You are the planner of a small research crew. Given a user's research "
        "topic, decompose it into 3 to 5 explicit, atomic subtasks. Each "
        "subtask must be answerable in 1-2 sentences by a researcher with web "
        "access. Output a numbered markdown list. Do not answer the subtasks "
        "yourself."
    ),
)


# ──────────────────────────────────────────────────────────────────────
# Module 2: Researcher (tool-using)
# ──────────────────────────────────────────────────────────────────────
RESEARCHER = Agent(
    name="researcher",
    role="Researcher",
    system=(
        "You are the researcher. You receive a numbered list of subtasks. For "
        "each subtask, decide whether to call the `web_search_stub` tool or "
        "answer directly. Cite the search snippet you used (in parentheses). "
        "Keep answers under 3 sentences each."
    ),
    can_use_tools=True,
)


# ──────────────────────────────────────────────────────────────────────
# Module 3: Critic
# ──────────────────────────────────────────────────────────────────────
CRITIC = Agent(
    name="critic",
    role="Critic",
    system=(
        "You are the critic. You receive the planner's subtasks and the "
        "researcher's answers. List specific weaknesses: missing evidence, "
        "unsupported claims, internal contradictions. Be terse and concrete. "
        "If everything is fine, reply `LGTM`."
    ),
)


# ──────────────────────────────────────────────────────────────────────
# Module 4: Writer
# ──────────────────────────────────────────────────────────────────────
WRITER = Agent(
    name="writer",
    role="Writer",
    system=(
        "You are the writer. You receive the original topic, the researcher's "
        "answers, and the critic's notes. Produce a tight markdown brief: a "
        "one-line summary, three to five bullet findings, and an honest "
        "`## Caveats` section reflecting the critic's weaknesses."
    ),
)


# ──────────────────────────────────────────────────────────────────────
# Crew (sequential)
# ──────────────────────────────────────────────────────────────────────
CREW: tuple[Agent, ...] = (PLANNER, RESEARCHER, CRITIC, WRITER)
