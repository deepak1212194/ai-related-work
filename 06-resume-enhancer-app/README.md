# AI Resume Enhancer — Multi-Agent Edition (v6)

> Upload a `.tex` resume, get back an ATS-optimised, Overleaf-ready `.tex` with per-section critic scores, a hiring-manager simulation, and JD keyword-match deltas — all powered by a free Groq API key.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![LLM: Groq (Free)](https://img.shields.io/badge/LLM-Groq%20Free-orange)

---

## Demo / Screenshot

```
[ Upload .tex ]  →  [ 8-stage pipeline with live progress ]  →  [ Download enhanced .tex ]
                                                                    ↓
                                                              [ Critic scores ]
                                                              [ JD match delta ]
                                                              [ Hiring-manager review ]
```

_Dark animated single-page UI with per-section before/after diffs, 5-dimension critic scores,
JD keyword match bars, ATS donut chart, and hiring-manager simulation panel._

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.10, 3.11, 3.12 |
| Groq API key | Free — get one at [console.groq.com/keys](https://console.groq.com/keys). No credit card required. |
| Hugging Face token (optional) | Free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Used as secondary extraction backend when `RESUME_ENABLE_MULTI_LLM=true`. |

---

## Quick Start — Local Python

**PowerShell:**

```powershell
git clone https://github.com/your-handle/ai-related-work.git
cd ai-related-work\06-resume-enhancer-app

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:GROQ_API_KEY = "gsk_..."          # paste your key
python -m ui.app
```

**Bash / macOS / Linux:**

```bash
git clone https://github.com/your-handle/ai-related-work.git
cd ai-related-work/06-resume-enhancer-app

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."          # paste your key
python -m ui.app
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Quick Start — Docker

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

docker compose up --build
```

The container maps port `7860` and hot-mounts `skills/` and `data/jds/` for live editing without a rebuild.

---

## Step-by-Step: Get a Groq Key and Run

1. Visit [console.groq.com/keys](https://console.groq.com/keys) and sign up (free, no credit card).
2. Click **Create API Key** and copy the `gsk_...` string.
3. Set the environment variable:
   - **PowerShell:** `$env:GROQ_API_KEY = "gsk_..."`
   - **Bash:** `export GROQ_API_KEY="gsk_..."`
   - **Docker:** add to `.env` as `GROQ_API_KEY=gsk_...`
4. Run `python -m ui.app` (or `docker compose up`).
5. Open the UI, go to the **Setup** tab — it should show "Groq: Ready".
6. Switch to the **Enhance** tab, upload your `.tex`, pick a target role, click **Enhance**.

---

## Available Groq Models

| Model ID | Free TPM | Notes |
|---|---|---|
| `llama-3.1-8b-instant` | 6,000 | Fastest; good for short resumes |
| `llama-3.3-70b-versatile` | 6,000 | Highest quality on Groq free tier |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 30,000 | **Recommended for large resumes** — 5x higher TPM headroom |

> **Auto-fallback:** When a model hits its TPM quota (HTTP 429) or has been decommissioned, the pipeline automatically rotates through the fallback list in order of TPM headroom. No manual intervention required. Set `GROQ_MODEL` to pin a specific model.

---

## Pipeline Diagram

```
Input .tex
     |
     v
+-------------+
|  1. PARSE   |  LLM-first (ExtractorAgent, 12K char window)
|             |  Regex fallback for any LLM failure
+------+------+
       |  ResumeIR
       v
+-------------+
|  2. REPAIR  |  ExtractorAgent.repair() — fills missed fields
|             |  Skipped when LLM-first extraction succeeded
+------+------+
       |
       v
+-------------+
| 3. COMPLETE |  CompleterAgent — inserts [PLACEHOLDER] tokens
|             |  deterministic; never guesses
+------+------+
       |
       v
+-------------+
|  4. PLAN    |  PlannerAgent — section ordering, lead-bullet hints
|             |  role-specific priority from skills/*.md
+------+------+
       |
       v
+-------------------------------------------+
|  5. ENHANCE (block-level batching)        |
|                                           |
|   for each experience / project block:   |
|     EnhancerAgent.draft_block()  ----+   |
|                                      v   |
|                               CriticAgent|
|                               .score_block()
|                                      |   |
|                         score >= 82? -+  |
|                         or iter >= 3?    |
|                                      v   |
|                         safe_apply() guard|
|                         (deterministic)  |
+------+------------------------------------+
       |  enhanced ResumeIR
       v
+-------------+
|  6. RENDER  |  Jinja2 + LaTeX template -> .tex string
+------+------+
       |
       v
+-------------+
|  7. SCORE   |  ATS keyword scan (deterministic)
|             |  JD match delta vs. curated JDs
+------+------+
       |
       v
+-------------+
| 8. REVIEW   |  RoleReviewerAgent — hiring-manager simulation
|             |  JSON: score, strengths, weaknesses, verdict
+-------------+
       |
       v
Enhanced .tex + scores + review
```

---

## Agents Reference

| Agent | Reads | Writes | Failure mode |
|---|---|---|---|
| **ExtractorAgent** | Raw `.tex` string (up to 12,000 chars) | Populated `ResumeIR` (LLM path) or repairs to existing IR | Falls back to regex parser; never crashes the pipeline |
| **CompleterAgent** | Partially-populated `ResumeIR` | `[PLACEHOLDER]` tokens for missing required fields | Deterministic; no LLM call; always succeeds |
| **PlannerAgent** | `ResumeIR` + role skill file | `SectionPlan` (ordered section list, lead-bullet index hints) | Falls back to default section order |
| **EnhancerAgent** | One section or block of bullets + role context | Rewritten plain-text bullet(s) via `draft()` or `draft_block()` | Returns empty string on LLM error; orchestrator keeps original |
| **CriticAgent** | Original + rewrite pair (or block of pairs) | JSON score dict: `scores`, `total`, `violations`, `fix_hint`, `verdict` | Returns conservative "iterate" fallback (non-final) or "accept" (final iteration) to prevent infinite loops |
| **IterativeOrchestrator** | Enhancer + Critic + `protected_terms` set | `SectionTrace` per bullet with full iteration history | Hard cap of 4 iterations; deterministic `safe_apply()` guard always runs last |
| **RoleReviewerAgent** | Full enhanced resume text + role profile | JSON: `overall_score`, `strengths`, `weaknesses`, `missing_keywords`, `one_line_verdict` | Skipped with warning; pipeline completes without it |
| **JDMatchAgent** | Resume text + curated JD keyword lists | `JDMatchReport`: per-JD scores, `avg_delta`, `top_gaps` | Deterministic (no LLM); skipped with warning on missing JD data |

---

## Safety Guarantees

1. **Never weakens.** The deterministic `safe_apply()` guard checks every rewrite against the original. If the rewrite shrinks to less than 50% of the original length, it is rejected and the original is kept.

2. **Never drops protected terms.** Protected terms are extracted dynamically from the user's own resume (every skill, framework, company name, certification, and proper noun). The baseline set additionally covers approximately 100 common ML/cloud/dev tools (PyTorch, FAISS, BERT, Azure, Kubernetes, etc.). Any rewrite that drops a protected term is rejected.

3. **Bounded iteration.** The Enhancer+Critic loop is hard-capped at 4 iterations regardless of settings. The `RESUME_MAX_ITERATIONS` env var controls the default (3); it cannot be set above 4.

4. **Never invents.** The core rules skill file (`skills/00_core_rules.md`) instructs the LLM to never fabricate metrics, percentages, years, employer names, product names, or certifications. The Critic agent penalises fabrication with up to -10 points per occurrence on the honesty dimension.

5. **Critic cannot escape the guard.** Even if the Critic scores a draft 100/100, the deterministic Python `safe_apply()` guard runs afterwards and can still reject the draft. The guard's decision is final and cannot be overridden by any LLM output.

---

## Configuration Reference

All settings are read from environment variables (or a `.env` file). Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | _(required for Groq)_ | Groq API key. Get free at console.groq.com/keys |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Active Groq model. Auto-rotates to fallbacks on 429 or decommission |
| `HF_API_KEY` | _(optional)_ | Hugging Face token for secondary extraction backend |
| `HF_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF model for enhancement (when backend=huggingface) |
| `RESUME_MAX_ITERATIONS` | `3` | Max Enhancer+Critic loop iterations per section (hard cap: 4) |
| `RESUME_ACCEPT_THRESHOLD` | `82` | Critic score at or above which a draft is accepted early |
| `RESUME_MAX_SECTION_CALLS` | `120` | Max total LLM calls for enhancement across all sections |
| `RESUME_LLM_TIMEOUT_S` | `90` | Per-call LLM timeout in seconds |
| `RESUME_MAX_UPLOAD_KB` | `512` | Maximum `.tex` upload size |
| `RESUME_MAX_RUNS_PER_HOUR` | `5` | Per-session rate limit (UI-level) |
| `RESUME_UI_PORT` | `7860` | Gradio server port (set via `--server-port` in launch args) |
| `RESUME_LOG_LEVEL` | `INFO` | Logging verbosity: DEBUG / INFO / WARNING / ERROR |
| `RESUME_LLM_MAX_RETRIES` | `3` | Retry attempts per LLM call on transient failure |
| `RESUME_MIN_DELTA` | `3` | Stop iterating if improvement between rounds is less than this value |
| `RESUME_WORK_TTL_HOURS` | `24` | Age in hours before job output directories are cleaned up |
| `RESUME_AUTH_USER` | _(empty)_ | Basic auth username (enable by setting both user and pass) |
| `RESUME_AUTH_PASS` | _(empty)_ | Basic auth password |
| `RESUME_ENABLE_MULTI_LLM` | `true` | Use cheaper HF model for extraction, Groq for enhancement |
| `RESUME_DEFAULT_ROLE` | `ai_ml_engineer` | Default role when none is selected in the UI |

---

## Project Layout

```
06-resume-enhancer-app/
├── app/
│   ├── agents/
│   │   ├── base.py            # Agent base class, clean_draft(), extract_json()
│   │   ├── extractor.py       # ExtractorAgent — LLM-first parse + repair
│   │   ├── completer.py       # CompleterAgent — deterministic placeholder fill
│   │   ├── planner.py         # PlannerAgent — section ordering
│   │   ├── enhancer.py        # EnhancerAgent — draft() + draft_block()
│   │   ├── critic.py          # CriticAgent — score() + score_block()
│   │   ├── orchestrator.py    # IterativeOrchestrator — loop + safety guard
│   │   ├── role_reviewer.py   # RoleReviewerAgent — hiring-manager simulation
│   │   └── jd_matcher.py      # JDMatchAgent — deterministic keyword scoring
│   ├── core/
│   │   ├── config.py          # Settings (env-driven, frozen dataclass)
│   │   ├── llm.py             # LLM backends: Groq, HuggingFace (+ retry logic)
│   │   ├── safety.py          # Deterministic rewrite guard
│   │   ├── skills.py          # Markdown skill-file loader (hot-reloadable)
│   │   ├── ir.py              # ResumeIR, PipelineResult, SectionTrace types
│   │   ├── ats.py             # ATS keyword scoring
│   │   └── context_budget.py  # Token estimation, relevant_keywords()
│   ├── parser/
│   │   └── tex_parser.py      # Heuristic regex .tex -> ResumeIR
│   ├── render/
│   │   └── template.tex.j2    # Jinja2 LaTeX output template
│   └── pipeline.py            # run_pipeline() — 8-stage orchestration
├── skills/                    # Hot-reloadable agent instruction files
│   ├── 00_core_rules.md       # Universal rules injected into every prompt
│   ├── 01_extraction.md       # ExtractorAgent instructions
│   ├── 02_completion.md       # CompleterAgent instructions
│   ├── 03_planning.md         # PlannerAgent instructions
│   ├── 04_enhancement.md      # EnhancerAgent instructions + examples
│   ├── 05_critique.md         # CriticAgent rubric + calibration examples
│   ├── 06_role_review.md      # RoleReviewerAgent instructions
│   ├── 07_jd_matching.md      # JDMatchAgent documentation
│   ├── role_ai_ml_engineer.md
│   ├── role_data_scientist.md
│   ├── role_software_engineer.md
│   ├── role_devops_cloud_engineer.md
│   └── role_product_manager.md
├── data/
│   └── jds/                   # Curated JD keyword JSON files per role
├── ui/
│   ├── app.py                 # FastAPI backend (SSE streaming, job queue)
│   └── index.html             # Single-page dark UI (animated, no framework)
├── tests/
│   ├── test_parser.py
│   ├── test_safety.py
│   └── test_moderncv.tex
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Customising Agent Behaviour

All agent instructions live in `skills/*.md` and are **hot-reloadable** — edit a file and the next pipeline run picks it up without restarting.

**File format:**

```markdown
# Title (ignored by the loader)

Default block content — used when no named block is requested.

## block_name

Named block content — agents request specific blocks by name to
keep prompt tokens tight.
```

**Common customisations:**

- **Change enhancement style** — edit `skills/04_enhancement.md`. Add new bullet examples under the `## bullet` heading.
- **Tune the critic rubric** — edit `skills/05_critique.md`. Adjust per-dimension scoring rules and calibration examples.
- **Add a new role** — create `skills/role_<role_id>.md` with sections `## priority_keywords`, `## hiring_signals`, `## red_flags`. Then register the role ID in `app/pipeline.py`'s `ROLES` dict.
- **Add diction rules** — extend the `## diction_blacklist` block in `skills/00_core_rules.md`.

---

## Deploying Live

### Hugging Face Spaces

```yaml
# README.md front matter for your Space
---
title: AI Resume Enhancer
sdk: docker
app_port: 7860
---
```

Set `GROQ_API_KEY` in the Space's **Secrets** panel. The app starts with no other configuration.

### Docker (any VPS or cloud VM)

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY and optional auth
docker compose up -d
```

Mount `skills/` and `data/jds/` as volumes (already configured in `docker-compose.yml`) so you can update instructions without rebuilding the image.

### Render / Railway

1. Connect your repository.
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python -m ui.app`
4. Add `GROQ_API_KEY` as an environment variable in the platform dashboard.
5. Set `RESUME_UI_PORT` to match the platform's expected port (Render: 10000, Railway: value of `$PORT`).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No LLM backend is configured" | Neither `GROQ_API_KEY` nor `HF_API_KEY` is set | Set `GROQ_API_KEY` in your `.env` or shell |
| Groq 429 rate limit errors | TPM quota exhausted for the active model | Handled automatically — the pipeline rotates to the next model in the fallback list. If all models are exhausted, wait 60 seconds. |
| "model decommissioned" | Groq retired a model ID | Handled automatically — the pipeline skips the decommissioned model and tries the next one |
| "File is too large" | `.tex` file exceeds `RESUME_MAX_UPLOAD_KB` (default 512 KB) | Increase `RESUME_MAX_UPLOAD_KB` or trim the `.tex` |
| "This .tex file does not look like a resume" | Uploaded a non-resume `.tex` (e.g. a paper or book) | Upload a resume `.tex` with name, experience, and skills sections |
| "Potentially dangerous LaTeX command" | File contains `\write18`, `\input{/`, or similar | Remove shell-escape and path-traversal commands from the `.tex` |
| Safety guard keeps rejecting rewrites | A protected term in the resume is very short or contains special characters | Set `RESUME_LOG_LEVEL=DEBUG` to see which term is failing in the output |
| Critic scores all show 0 | LLM returned non-JSON on the critic call | Transient — retry the run. If persistent, check that `GROQ_MODEL` is a valid active model |
| UI shows "You've used all N runs" | Per-hour rate limit reached | Wait for the window to reset (shown in the message), or increase `RESUME_MAX_RUNS_PER_HOUR` |
| Docker container exits immediately | Missing env var or port conflict | Check `docker compose logs resume-enhancer` for the specific error |

---

## Why This Architecture

### Enhancer+Critic vs a Larger Crew

A tempting alternative is a large crew of specialised agents (one per section type, one for ATS, one for format). In practice, the overhead of coordinating many agents across a multi-section resume produces more failure modes than it solves. The Enhancer+Critic two-agent loop with a bounded iteration count is simple, auditable, and produces consistent quality. The Critic's structured JSON output gives precise feedback the Enhancer can act on, and the deterministic safety guard provides a hard correctness floor that no agentic reasoning layer can violate.

### Block-Level Batching Rationale

Naive implementations call the LLM once per bullet. A typical resume has 20–40 bullets, leading to 60–120 LLM calls when iterated 3 times. Block-level batching (`EnhancerAgent.draft_block()`) sends all bullets from one experience or project block in a single call, reducing LLM calls by 3–6x for typical resumes. The `[N]` indexed output format is deterministic to parse, and the orchestrator falls back transparently to per-bullet calls if the block response is malformed or returns fewer than half the expected lines.

---

## License

MIT. See [LICENSE](../LICENSE) for details.
