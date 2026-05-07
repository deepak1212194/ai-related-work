# 06 · Resume Enhancer — 2-Agent Skill-Driven Loop

A **skill-file-driven, two-agent resume enhancement service**. An Enhancer
agent drafts each section using rules loaded from editable markdown files;
a Critic agent grades the draft on a 5-dimension, 100-point rubric; an
Orchestrator iterates up to 3 times with early-stop. Final output is gated
by deterministic Python guards (length ratio + protected-term-drop check)
so the agentic loop can never weaken or fabricate.

Upload a `.pdf` or `.tex`, pick a target role, get back a polished
`.tex` + `.pdf` plus an ATS keyword score and a per-section iteration trace.

---

## What it demonstrates

- **2-agent hybrid pattern** — Enhancer (drafts) + Critic (rubric scorer
  returning JSON) inside a bounded iteration loop. Real agentic behaviour
  applied where it earns its keep, instead of a 4-agent crew built for show.
- **Skill-file architecture** — every rule, role profile, and per-section
  task template lives in `skills/*.md`. Edit the files, hot-reload via
  `/api/skills/reload`, no code changes.
- **Deterministic safety net** — the agentic loop runs *inside* Python
  guards (length ≥ 50% of input; no protected term dropped). The critic
  cannot accept a draft that violates these.
- **Bounded everything** — per-call timeout, per-section budget, overall
  job timeout, max-iterations cap. The service cannot loop forever.
- **Iteration traces** — every section returns its draft history with
  per-dimension scores and the critic's violations. Visible in the UI.
- **Multi-backend** — Hugging Face (default) or Anthropic Claude with
  prompt caching on the system prompt.

---

## Architecture

```
                         skills/
                           core_rules.md
                           role_*.md   (5 roles)
                           section_tasks.md
                                │
                  ┌─────────────┴─────────────┐
                  │      Skill Loader         │  ← hot-reloadable
                  └─────────────┬─────────────┘
                                ▼
            ┌───────────────────────────────────────┐
            │  IterativeOrchestrator (per section)  │
            │                                       │
            │   ┌──────────────┐  draft  ┌────────┐ │
            │   │ EnhancerAgent│────────▶│ Critic │ │
            │   │              │         │ Agent  │ │
            │   │              │◀──fix───│ (JSON) │ │
            │   └──────────────┘         └────────┘ │
            │                                       │
            │   max 3 iter · accept ≥ 80 · Δ < 3    │
            └───────────────┬───────────────────────┘
                            ▼
            ┌───────────────────────────────────────┐
            │  Deterministic Python guards          │
            │   - length ratio ≥ 50 %               │
            │   - no protected term dropped         │
            └───────────────┬───────────────────────┘
                            ▼
                    .tex  +  .pdf  +  ATS score
                                  +  iteration trace
```

### Critic rubric (5 dims × 20 = 100)

| Dimension | What it grades |
|---|---|
| `honesty` | Every fact in the rewrite traces to the input — no fabricated metrics, names, or dates. |
| `action_verb` | Senior past-tense lead (Architected / Engineered / Designed / Owned / Shipped). Penalises "Worked on", "Helped with", "Responsible for". |
| `specificity` | Names a specific system, scope, technique, or stack — not vague. |
| `tightness` | No filler, no hedging. Bullets fit ≤ 360 chars. |
| `keyword_retention` | Every tech term, number, and proper noun from the input still appears. |

The critic returns a **JSON object** (parsed robustly with regex
fallbacks). On any parse / LLM failure the orchestrator degrades to
single-call behaviour — never a hard error.

### Why two agents and not one shot?

Subjective dimensions (verb-tier, scope-phrase appropriateness,
"differentiation") are exactly what an LLM is good at *judging*. A draft +
critique loop catches misses a single shot can't — e.g. draft #1 leads
with "Worked on", critic flags the action-verb violation, draft #2 leads
with "Architected".

### Why not full multi-agent (CrewAI / 4-agent crew)?

Resume rewriting is a focused task with an objective rubric. A 4-agent
crew (Researcher + Writer + Critic + Editor) would add latency, cost, and
indeterminism without lifting quality. Two specialised agents + a
deterministic guard layer hits a clean spot on the cost / quality curve.

---

## Quick start

```bash
cd 06-resume-enhancer-app
pip install -r requirements.txt

# Default backend: Hugging Face
export HF_API_KEY=hf_...
uvicorn app.main:app --reload --port 8006

# Or Claude (system prompt is cached across section calls)
export RESUME_LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-...
uvicorn app.main:app --reload --port 8006

# Tune the agent loop (all optional)
export RESUME_AGENT_CRITIC_ENABLED=true        # default
export RESUME_AGENT_MAX_ITERATIONS=3           # 1-5
export RESUME_AGENT_ACCEPT_THRESHOLD=80        # 0-100
export RESUME_AGENT_MIN_DELTA_TO_CONTINUE=3    # early-stop sensitivity

# Open http://localhost:8006
```

Set `RESUME_AGENT_CRITIC_ENABLED=false` to fall back to single-call
behaviour (no extra LLM hits) — useful for benchmarking the lift the
critic loop provides.

---

## Customising the agents

Every behaviour the agents exhibit is in markdown:

```bash
# Change core enhancement rules (applies to every role)
vim skills/core_rules.md

# Change per-role emphasis / priority keywords
vim skills/role_ai_ml_engineer.md

# Change per-section task templates (Summary / Bullet / Skills / ...)
vim skills/section_tasks.md

# Add a new role profile
cp skills/role_software_engineer.md skills/role_blockchain_engineer.md
$EDITOR skills/role_blockchain_engineer.md

# Hot-reload — no restart needed
curl -X POST http://localhost:8006/api/skills/reload
```

The Critic prompt itself (the rubric the Critic uses) lives in
`app/agents.py :: CRITIC_SYSTEM_PROMPT` because its JSON output schema
is parsed by code; changing that requires a code change.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status, backend, pdflatex, skill files loaded |
| `GET` | `/api/roles` | Available role profiles |
| `GET` | `/api/skills` | Skill file loading status |
| `POST` | `/api/skills/reload` | Hot-reload skill files |
| `POST` | `/api/enhance` | Upload + enhance → .tex / .pdf + ATS score + iteration trace |
| `POST` | `/api/ats-score` | Standalone ATS keyword scoring for any text |
| `GET`  | `/api/result/{id}.tex` | Download enhanced .tex |
| `GET`  | `/api/result/{id}.pdf` | Download compiled .pdf |

### Iteration trace (sample)

The `/api/enhance` response now includes per-section `iterations[]`:

```json
{
  "previews": [
    {
      "section": "Bullet · Senior AI Engineer",
      "before": "Worked on RAG pipeline using FAISS and PyTorch.",
      "after":  "Architected a production RAG pipeline — the retrieval surface for the legal-search product. Built FAISS-backed semantic retrieval over 1.2M chunks with PyTorch-served re-rankers and idempotent ingest.",
      "changed": true,
      "final_score": 92,
      "iterations_used": 2,
      "iterations": [
        {
          "iteration": 1,
          "draft": "Built RAG pipeline with FAISS and PyTorch...",
          "score": 64,
          "dim_scores": {"honesty": 20, "action_verb": 8, "specificity": 8, "tightness": 14, "keyword_retention": 14},
          "violations": ["lead verb 'Built' is acceptable but not architect-tier"],
          "verdict": "iterate",
          "accepted": false
        },
        {
          "iteration": 2,
          "draft": "Architected a production RAG pipeline...",
          "score": 92,
          "verdict": "accept",
          "accepted": true
        }
      ]
    }
  ]
}
```

---

## Safety guarantees

1. **Never weakens** — the deterministic length guard rejects shrinkage.
2. **Never fabricates** — protected terms (model names, frameworks, named
   numbers) are checked at the regex level after the critic accepts.
3. **Bounded iteration** — `max_iterations` (default 3) + per-section
   time budget + overall job timeout. The loop cannot run forever.
4. **Critic can fail safely** — JSON parse errors, LLM timeouts, and
   schema violations all degrade to "accept current draft", never to
   a hard failure.
5. **Structured errors** — every code path returns an `EnhanceResponse`
   with `status="error"` and a human-readable warning, never a 500.

---

## Stack

FastAPI · Pydantic · Jinja2 · PyMuPDF · pdflatex · Hugging Face / Anthropic
· Docker
