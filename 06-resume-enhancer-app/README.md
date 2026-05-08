# 06 · Resume Enhancer — Multi-Agent Edition (v4)

A production-grade, multi-agent, skill-driven resume rewriter. Upload a `.tex`
resume, pick a target role, get back an Overleaf-ready `.tex` plus a critic
trace, hiring-manager simulation across 5 roles, and JD keyword scoring
against 25 curated job descriptions.

> Input: `.tex` · Output: `.tex` · UI: Gradio · LLM: Claude Code login / Anthropic API / Hugging Face · Deploy: Docker.

---

## Authentication options (pick whichever is easiest)

The app **auto-detects** whichever backend is configured. You don't need
all three. In order of how-easy-to-set-up:

### 1. Claude Code login — zero API key

If you already use Claude Code (the CLI), you're already done.

```bash
pip install claude-agent-sdk     # already in requirements.txt
claude /login                     # one-time, opens a browser
```

The app picks up your subscription automatically. No `ANTHROPIC_API_KEY` needed.

### 2. Anthropic API — highest quality, paid per token

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # macOS / Linux
$env:ANTHROPIC_API_KEY='sk-ant-...'      # PowerShell
```

### 3. Hugging Face — free tier

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export HF_API_KEY=hf_...
```

The **Setup** tab in the UI shows live status of all three and walks
through whichever you haven't configured yet.

---

## What it does end-to-end

```
.tex input
   |
   v
+--------------+    +-------------+    +-------------+    +-------------+
| Parse (regex)| -> | Repair (LLM)| -> | Complete    | -> | Plan (LLM)  |
| -> ResumeIR  |    | (Extractor) |    | placeholders|    | section order
+--------------+    +-------------+    +-------------+    +-------------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | Per-section loop     |
                                                   | Enhancer <-> Critic  |
                                                   | (max 3 iter, >=82)   |
                                                   +----------------------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | Safety guards        |
                                                   | length + protected   |
                                                   +----------------------+
                                                              |
                          +-----------------------------------+--------------------+
                          v                                                        v
              +-------------------+                                  +-------------------+
              | Render -> .tex    |                                  | ATS / JD / Review |
              | (Jinja2 template) |                                  | across 5 roles    |
              +-------------------+                                  +-------------------+
```

### The agents

| Agent | Reads | Writes | Failure mode |
|---|---|---|---|
| `ExtractorAgent` | parsed IR + raw .tex | repaired IR | falls through to parser output |
| `CompleterAgent` | IR | IR with `[PLACEHOLDER]` tokens for missing required fields | always succeeds (deterministic) |
| `PlannerAgent` | IR + role profile | section / skill / experience ordering | falls back to per-role default |
| `EnhancerAgent` | one section + critique | rewritten section | returns "" → orchestrator keeps original |
| `CriticAgent` | (before, after) | 5-dim score + verdict (JSON) | permissive accept on parse error |
| `IterativeOrchestrator` | the two above | bounded loop, safety guard | deterministic guard rejects bad rewrites |
| `RoleReviewerAgent` | full enhanced resume + role profile | strengths / weaknesses / verdict | one-line "(unavailable)" on failure |
| `JDMatchAgent` | resume text + curated JDs | per-JD score, delta, top gaps | deterministic — never errors |

### Safety guarantees

1. **Never weakens.** A safety guard rejects any rewrite that shrinks below
   50% of the original or grows beyond 4× the original.
2. **Never drops a protected term.** Frameworks, models, services, and
   numbers from the input must appear in the output. Regex-checked.
3. **Bounded iteration.** Max 4 iterations per section, hard cap. Per-section
   time budget. Job-level deadline.
4. **Never invents.** Required fields the input lacks become visible
   `[PLACEHOLDER]` tokens — the user replaces them before sending.
5. **Critic can't escape the guard.** Even if the critic accepts a draft
   that violates a rule, the deterministic guard has the final word.

---

## Quick start

### Locally (Python)

```bash
cd 06-resume-enhancer-app
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Quickest path - no API key needed if Claude Code is installed:
claude /login

python -m ui.gradio_app
# UI at http://localhost:7860
```

### With Docker

```bash
docker compose --env-file .env up --build
# UI at http://localhost:7860
```

### Smoke test (no LLM key required)

```bash
python -m tests.smoke_parse      # parse + render a sample .tex
```

---

## Deploying live

The app is just a Gradio server in a slim Docker image — every Docker host
works. Two recommended targets:

### Hugging Face Spaces (free)

```
gradio deploy
```

…or use the "Duplicate this Space" button after pushing a `Dockerfile` Space.

### Render / Railway / Fly.io

Point them at this directory; the `Dockerfile` is the entry. Set
`ANTHROPIC_API_KEY` (or `HF_API_KEY`) as a secret and you're live.

---

## Project layout

```
06-resume-enhancer-app/
├── app/
│   ├── core/         IR · config · LLM · safety · skills · ATS
│   ├── agents/       Extractor · Completer · Planner · Enhancer · Critic
│   │                 RoleReviewer · JDMatcher · Orchestrator
│   ├── parser/       .tex -> ResumeIR (deterministic)
│   ├── render/       Jinja2 template + ResumeIR -> .tex
│   └── pipeline.py   top-level orchestrator
├── skills/           markdown rules (hot-reloadable)
│   ├── 00_core_rules.md
│   ├── 01_extraction.md ... 07_jd_matching.md
│   └── role_*.md     (5 role profiles)
├── data/jds/         curated JDs (5 roles × 5 JDs)
├── ui/               gradio_app.py
├── examples/         sample input .tex
├── tests/            smoke tests
├── Dockerfile · docker-compose.yml · requirements.txt · .env.example
└── README.md
```

---

## Customising agent behaviour

Every behaviour the agents exhibit is in markdown:

```bash
# Change the global rules every section follows
$EDITOR skills/00_core_rules.md

# Tune the priority keywords for AI/ML Engineer role
$EDITOR skills/role_ai_ml_engineer.md

# Adjust the critic rubric thresholds
$EDITOR skills/05_critique.md

# Add a new role
cp skills/role_software_engineer.md skills/role_blockchain_engineer.md
$EDITOR skills/role_blockchain_engineer.md
```

The skill loader picks up edits on each pipeline run — no restart needed.

---

## Configuration

All env vars are optional (sensible defaults). See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `RESUME_LLM_BACKEND` | `auto` | `auto` / `claude_code` / `anthropic` / `huggingface` |
| `ANTHROPIC_API_KEY` | — | required only for `anthropic` backend |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | latest Sonnet works well |
| `HF_API_KEY` | — | required only for `huggingface` backend |
| `HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | any HF Inference-API model |
| `RESUME_MAX_ITERATIONS` | `3` | per-section loop cap (hard cap 4) |
| `RESUME_ACCEPT_THRESHOLD` | `82` | critic total / 100 to accept early |
| `RESUME_LOG_LEVEL` | `INFO` |  |

`claude_code` backend has no env vars — it uses your `claude /login` session.

---

## Why two-agent (Enhancer + Critic) and not a 4-agent crew?

Resume rewriting is a focused task with an objective rubric. A 4-agent crew
(Researcher + Writer + Critic + Editor) would add latency, cost, and
indeterminism without lifting quality. Two specialised agents + a deterministic
guard layer hits a clean spot on the cost / quality curve.

The role-reviewer adds a *separate* loop (one call per role) which is a
**different task** — it's a hiring-manager simulator, not part of the
rewrite loop. That's why it's a distinct agent.

---

## License

MIT for the application code. The LaTeX template in `app/render/template.tex.j2`
is adapted from [Jake's Resume](https://github.com/jakegut/resume) (MIT).
