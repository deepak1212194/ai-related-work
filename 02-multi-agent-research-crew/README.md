# 02 · Multi-Agent Research Crew

A small, runnable **multi-agent system** that splits a research-style question into four sequentially-coordinated specialists:

```
   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
   │ Planner  │ →  │  Researcher  │ →  │  Critic  │ →  │  Writer  │
   └──────────┘    └──────────────┘    └──────────┘    └──────────┘
        │                │                  │                │
        ▼                ▼                  ▼                ▼
   subtask plan    tool-using calls    weakness pass    final brief
```

The crew is intentionally **simple** — four agents, sequential hand-off, two tools, one shared context. The architecture is deliberately the same shape used in production multi-agent systems but stripped down to what runs from a single CLI.

## Why this shape

Most "agentic AI" demos use a single model talking to itself in a loop. That works for toy tasks but fails on anything compositional because there's no division of responsibility. A small, well-defined crew with:

- **A planner** that turns the input into 3–5 explicit subtasks,
- **A researcher** that is the only agent allowed to call tools,
- **A critic** that adversarially reviews and flags gaps,
- **A writer** that produces the final markdown brief,

…tends to generalise much better. Each agent has a tightly-scoped system prompt, which keeps the model on rails.

## Quick start

```bash
cd 02-multi-agent-research-crew
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# OPENAI_API_KEY enables real generation; without it the crew runs in
# offline-stub mode and prints the orchestration trace.
export OPENAI_API_KEY=sk-...   # optional

python -m src.main --topic "What are the key trade-offs of MoE LLMs vs dense models?"
```

## Project layout

```
02-multi-agent-research-crew/
├── src/
│   ├── agents.py       # Agent definitions (system prompts + roles)
│   ├── tools.py        # Tool functions: web_search_stub, calculator
│   ├── orchestrator.py # Sequential coordinator with shared context
│   └── main.py         # CLI entry
├── examples/
│   └── sample_run.md   # Sample transcript
└── requirements.txt
```

## Notes

- The `web_search_stub` returns canned text so the demo is fully deterministic and offline-runnable. Swap in a real search tool (Tavily, SerpAPI, etc.) for live use.
- The orchestrator is hand-written rather than using a framework — this is intentional. It makes the data flow inspectable and the failure modes explicit.
- All system prompts live in `agents.py` so prompt-tuning is one-file edits.
