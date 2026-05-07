# 06 · Resume Enhancer App

A FastAPI service that takes a resume — **PDF or LaTeX** — chooses one of **5 target roles**, and returns a polished `.tex` + compiled `.pdf` that applies the rules from the [`senior-ai-resume-craft`](../) skill.

> **Restrictive by design.** The enhancer only ever lifts the resume — it cannot weaken, remove a fact, drop a tech keyword, or fabricate a metric. **Crash-proof by design** — any internal failure returns a structured response explaining what happened; the API never throws a 500 to the user.

## Highlights

- 🎯 **5 role profiles** out of the box — pick the target audience and the LLM gets role-specific emphasis
- 📄 **PDF or .tex input** — PyMuPDF section-aware parser for PDFs; regex-based section extraction for LaTeX
- 🤖 **Two LLM backends** — Hugging Face (`Llama-3.1-8B-Instruct`) by default; swap to Anthropic (`claude-opus-4-7` with adaptive thinking + prompt caching) by setting `RESUME_LLM_BACKEND=anthropic`
- 🛡️ **Three layers of safety guards** — per-LLM-call timeout, length-ratio floor, protected-term-drop check
- ⏱️ **Three layers of bounded execution** — per-call timeout (30 s), max bullets to enhance (30), overall job timeout (3 min). The app cannot enter an infinite loop.
- 👀 **Before/after preview per section** — collapsible cards in the UI showing exactly what changed; "kept (reason)" notes when an enhancement was rejected
- 🎨 **Polished UI** — role pills, animated loading messages, progress bar, structured error states, dark/light auto theme
- 🐳 **Dockerised** with TeX Live preinstalled for one-command PDF output

## The 5 target roles

| Role | What the LLM emphasizes |
|---|---|
| **AI / ML Engineer** | LLMs, RAG, multi-agent, fine-tuning, model evaluation, MLOps |
| **Software Engineer** | Distributed systems, scale, p99 latency, languages, architectural decisions |
| **Data Scientist** | Business impact metrics, A/B testing, statistical rigor, BI tools |
| **Product Manager** | Launches with quantified outcomes, OKRs, cross-functional leadership |
| **DevOps / Cloud Engineer** | DORA metrics, IaC, observability, infrastructure scale |

The role list is fetched live by the UI from `/api/roles`, so adding a new role only needs an entry in `app/rules.py` → `ROLE_PROFILES`.

## Architecture

```
Upload (PDF / .tex)  +  Role pick
        │
        ▼
parser.py  →  ParsedResume
        │
        ▼
enhancer.py  ──────────────►  per-section calls (with hard limits)
        │                        │
        │                        ▼
        │                     llm.py
        │                       ├── HuggingFaceClient (default; per-call timeout)
        │                       └── AnthropicClient (swap-in; cached system prompt)
        │                        │
        │                        ▼
        │                     rules.py
        │                       ├── SYSTEM_RULES (universal "no-degrade")
        │                       └── ROLE_PROFILES[role].emphasis (role-specific)
        ▼
compiler.py  ─►  Jinja(base.tex)  ─►  pdflatex  ─►  resume.tex + resume.pdf
        │
        ▼
EnhanceResponse with: tex_path, pdf_path, notes[], previews[], warnings[]
```

## Project layout

```
06-resume-enhancer-app/
├── app/
│   ├── main.py            # FastAPI: /api/enhance, /api/result, /api/roles, /health
│   ├── config.py          # All hard limits (timeouts, max bullets, max upload)
│   ├── schemas.py         # ParsedResume, EnhanceResponse, RoleInfo, SectionPreview
│   ├── llm.py             # LLMClient ABC; HuggingFace + Anthropic implementations
│   ├── rules.py           # SYSTEM_RULES + 5 ROLE_PROFILES + per-section task prompts
│   ├── parser.py          # parse_pdf() · parse_tex()
│   ├── enhancer.py        # Orchestrator with timeouts + safety guards
│   └── compiler.py        # Jinja(base.tex) → pdflatex
├── templates/
│   └── base.tex           # Canonical Jake-style LaTeX with custom Jinja delimiters
├── ui/
│   └── index.html         # Role pills + drag-drop + before/after previews
├── data/jobs/             # Per-job output (gitignored)
├── Dockerfile             # python:3.11-slim + texlive-latex-recommended
├── docker-compose.yml
└── requirements.txt
```

## Run with Docker (recommended)

```bash
cd 06-resume-enhancer-app

# Default — Hugging Face backend
HF_API_KEY=hf_...   docker compose up --build

# Or — Claude backend (uses Opus 4.7 with prompt caching on the rules)
RESUME_LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-...   docker compose up --build

open http://localhost:8005/                 # macOS / Linux
start http://localhost:8005/                # Windows
```

## Run locally (without Docker)

```bash
cd 06-resume-enhancer-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Local pdflatex required for PDF output (otherwise only .tex is returned).
# Ubuntu:  sudo apt-get install texlive-latex-recommended texlive-fonts-extra
# macOS:   brew install --cask basictex && eval "$(/usr/libexec/path_helper)"
# Windows: install MiKTeX or TeX Live; ensure pdflatex is on PATH.

export HF_API_KEY=hf_...
uvicorn app.main:app --reload --port 8005
```

## API reference

```
GET  /                           drag-drop UI
GET  /health                     {backend, backend_configured, pdflatex_available, available_roles[]}
GET  /api/roles                  the 5 role profiles
POST /api/enhance                multipart upload + role → EnhanceResponse
GET  /api/result/{job_id}.tex    download enhanced .tex
GET  /api/result/{job_id}.pdf    download enhanced .pdf (404 if compile failed)
```

`POST /api/enhance` form fields:

| field | type | required | notes |
|---|---|---|---|
| `file` | multipart file | ✅ | `.pdf` or `.tex`, max 5 MB |
| `role` | string | optional | one of the 5 role IDs from `/api/roles`; defaults to `ai_ml_engineer` |

Full OpenAPI schema at `/docs` and `/redoc`.

## How the safety net works

The app has **three independent layers** that prevent infinite loops, hangs, or crashes:

| Layer | What it does | Where |
|---|---|---|
| **1. Per-call timeout** | Every LLM API call has a hard `timeout` parameter. If the LLM hangs, the call raises and the original section is preserved. Default 30 s. | `app/config.py: llm_call_timeout_seconds` |
| **2. Loop bounds** | The bullet-enhancement loop is hard-capped at `max_bullets_to_enhance` (default 30). Skills buckets are capped at `max_skills_buckets_to_enhance` (default 10). Anything beyond is preserved verbatim with a warning. | `app/config.py` |
| **3. Overall wall clock** | `enhancer.enhance()` checks elapsed time before every section call. If the total exceeds `overall_job_timeout_seconds` (default 180 s), remaining sections are kept as-is and a warning is added. | `app/enhancer.py` |

Plus, **the entire `enhance_resume` endpoint is wrapped in fault handlers** — every failure mode (parse error, LLM error, template error, compile error, oversized file, empty file, unsupported format) returns a structured `EnhanceResponse` with `status="error"` and a human-readable warning. The user always sees a useful message, never a stack trace.

And on every accepted run, two **client-side guards** run on the LLM output before it's used:

| Guard | When it triggers | Action |
|---|---|---|
| **Length-ratio** | Candidate < 50% of input length | Keep input, append note "output too short" |
| **Protected-term-drop** | Input had `RAG`, `FAISS`, a YOLO version, a model name, an R² number, etc. — and the candidate dropped it | Keep input, append note `"would drop protected terms: …"` |

Every section's outcome is reported in the API response and rendered as a collapsible card in the UI, so users see exactly what was enhanced vs. preserved and why.

## Configuration

All env vars use the `RESUME_` prefix.

| Variable | Default | Notes |
|---|---|---|
| `RESUME_LLM_BACKEND` | `huggingface` | `huggingface` or `anthropic` |
| `RESUME_HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Any chat model on HF Inference |
| `HF_API_KEY` | *(unset)* | Required for Llama-family models |
| `RESUME_ANTHROPIC_MODEL` | `claude-opus-4-7` | Best-quality Claude for editing |
| `ANTHROPIC_API_KEY` | *(unset)* | Required when backend=anthropic |
| `RESUME_DEFAULT_ROLE` | `ai_ml_engineer` | Used when the request omits a role |
| `RESUME_LLM_CALL_TIMEOUT_SECONDS` | `30` | Hard timeout per LLM call |
| `RESUME_OVERALL_JOB_TIMEOUT_SECONDS` | `180` | Hard wall clock per resume |
| `RESUME_MAX_BULLETS_TO_ENHANCE` | `30` | Bullets beyond this cap are kept verbatim |
| `RESUME_MAX_SKILLS_BUCKETS_TO_ENHANCE` | `10` | Same idea, for skills lines |
| `RESUME_PDFLATEX_CMD` | `pdflatex` | Set to `tectonic` if installed instead |
| `RESUME_MAX_UPLOAD_SIZE_MB` | `5` | Per-file upload cap |

## Limits and honest constraints

- **PDF parsing is heuristic.** Highly stylized PDFs (multi-column, embedded images, scanned pages) may yield empty or partial sections. The app returns a clear error in that case — try uploading the `.tex` instead.
- **The compiled `.pdf` uses one canonical template** (Jake-style). The goal is a clean senior-grade output, not pixel-perfect round-tripping of arbitrary input styles. Heavy custom formatting in the input will not be preserved — but every line of content will be.
- **No metric fabrication.** If the input doesn't contain a number, the enhanced output won't either. The LLM is restrictively prompted and verified by the protected-term guard.
- **No silent failures.** If a section can't be improved (LLM hung, output too short, dropped a keyword), the input is kept and the reason is shown in the response. Users always know what happened.
