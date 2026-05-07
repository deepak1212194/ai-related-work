# 06 · Resume Enhancer App

A FastAPI service that takes a resume — **PDF or LaTeX** — and returns an enhanced version as **both** a polished `.tex` source file **and** a compiled `.pdf`. It applies the rules from the [`senior-ai-resume-craft`](../) skill: stronger action verbs, italic em-dash scope phrases, ATS keyword preservation, corporate-confidentiality safety, and a strict no-degradation guarantee.

> **Restrictive by design.** The enhancer only ever lifts the resume — it cannot weaken, remove a fact, drop a tech keyword, or fabricate a metric. Every section's output is length-checked and keyword-checked against the input, and the original is preserved if the candidate output is suspect.

## Highlights

- **Two LLM backends, one interface** — defaults to **Hugging Face** (`Llama-3.1-8B-Instruct` via the HF Inference API, free tier-friendly). Swap to **Anthropic Claude** (`claude-opus-4-7` with adaptive thinking + prompt-caching on the rules system prompt) by setting `RESUME_LLM_BACKEND=anthropic`.
- **PDF or .tex input** — PyMuPDF parses PDFs by section heuristics; .tex files are split by `\section{}` blocks.
- **Section-by-section enhancement** — summary, skills, experience bullets, and achievements each get their own focused prompt with their own length / keyword guard.
- **Canonical Jake-style LaTeX output** — Jinja-rendered template with custom delimiters (`<<` / `>>`) so we never collide with LaTeX's `{}`.
- **PDF compilation** — runs `pdflatex` if available; the `.tex` is always returned even if compilation isn't available locally.
- **Drag-drop UI** at `/` with live status of the LLM backend and pdflatex availability.
- **Dockerised** with TeX Live preinstalled — one command up and PDF output works out of the box.

## Architecture

```
       upload                                              .tex + .pdf
   PDF ──────▶ /api/enhance ──▶ parser ──▶ enhancer ──▶ compiler ──▶
   .tex                          (PDF /     (LLM        (Jinja
                                  .tex)     guards)      → pdflatex)

                                            │
                                            ▼
                                    ┌───────────────┐
                                    │ rules.py      │
                                    │  • SYSTEM     │
                                    │  • BULLET     │
                                    │  • SUMMARY    │
                                    │  • SKILLS     │
                                    │  • ACHIEVEMENT│
                                    └───────┬───────┘
                                            │
                                            ▼
                              llm.py (HF default · Claude swap)
```

## Project layout

```
06-resume-enhancer-app/
├── app/
│   ├── main.py            # FastAPI: /api/enhance, /api/result, /health
│   ├── config.py          # pydantic-settings (RESUME_*  env vars)
│   ├── schemas.py         # ParsedResume, EnhanceResponse, …
│   ├── llm.py             # LLMClient ABC · HuggingFaceClient · AnthropicClient
│   ├── rules.py           # System prompt + per-section task prompts
│   ├── parser.py          # parse_pdf() · parse_tex()
│   ├── enhancer.py        # orchestrates LLM calls with safety guards
│   └── compiler.py        # render_tex(parsed) · compile_pdf(tex)
├── templates/
│   └── base.tex           # Jake-style canonical resume template (Jinja)
├── ui/
│   └── index.html         # drag-drop upload + result downloads
├── data/jobs/             # per-job working dir (gitignored)
├── Dockerfile             # python:3.11-slim + texlive-latex-recommended
├── docker-compose.yml
└── requirements.txt
```

## Run with Docker (recommended — TeX Live preinstalled)

```bash
cd 06-resume-enhancer-app

# Default: Hugging Face backend (Llama 3.1 8B)
HF_API_KEY=hf_...   docker compose up --build

# Or: Claude backend
RESUME_LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-...   docker compose up --build

open http://localhost:8005/                 # macOS / Linux
start http://localhost:8005/                # Windows
```

## Run locally (without Docker)

```bash
cd 06-resume-enhancer-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Local pdflatex required for PDF output. On Ubuntu:
#   sudo apt-get install texlive-latex-recommended texlive-fonts-extra
# On macOS (BasicTeX):
#   brew install --cask basictex && eval "$(/usr/libexec/path_helper)"
# On Windows: install MiKTeX or TeX Live; ensure `pdflatex` is on PATH.

export HF_API_KEY=hf_...
uvicorn app.main:app --reload --port 8005
```

## API reference

```
GET  /                          → drag-drop UI
GET  /health                    → readiness ({backend, backend_configured, pdflatex_available})
POST /api/enhance               → multipart upload → {job_id, tex_path, pdf_path, notes[]}
GET  /api/result/{job_id}.tex   → download enhanced .tex
GET  /api/result/{job_id}.pdf   → download enhanced .pdf (404 if compilation failed)
```

OpenAPI schema auto-generated at `/docs` and `/redoc`.

## The enhancement rules (from the skill)

The system prompt baked into [`app/rules.py`](./app/rules.py) is the load-bearing part. Excerpt:

> 1. Never remove a specific fact, number, model name, library, or tech keyword present in the input.
> 2. Never weaken a claim. Only strengthen.
> 3. Never invent metrics, scale claims, or achievements not in the input.
> 4. Replace passive openers with senior verbs: Architected, Engineered, Owned, Designed, Co-invented, Productionized, Shipped, Migrated.
> 5. Lead with WHAT in plain English; follow with HOW + tech keywords.
> 6. Preserve italic em-dash scope phrases when supported by the input.
> 7. Maintain ATS keyword density (RAG, fine-tuning, multi-agent, etc.).
> 8. Tighten bloat ("Built and shipped" → "Shipped") without losing information.
> 9. Never change company / product / employer / dates / institution names.
> 10. Output only the rewritten text — no preamble, no commentary, no markdown fences.
>
> If unsure how to enhance a section, return the input unchanged. **Never output a worse version.**

In addition, `enhancer.py` runs two **client-side safety guards** on every LLM response:

| Guard | Triggers when | Action |
|---|---|---|
| Length-ratio | Candidate is < 50% of input length | Keep input |
| Protected-term drop | Candidate is missing a term that was in the input (e.g. "FAISS", "RAG", a model name, a numeric metric) | Keep input |

A note is appended for every section so you can see what was enhanced vs kept.

## Configuration

All env vars use the `RESUME_` prefix.

| Variable | Default | Notes |
|---|---|---|
| `RESUME_LLM_BACKEND` | `huggingface` | `huggingface` or `anthropic` |
| `RESUME_HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Any chat-completion model on HF Inference |
| `HF_API_KEY` | *(unset)* | Required for Llama-family models |
| `RESUME_ANTHROPIC_MODEL` | `claude-opus-4-7` | Best-quality Claude for editing tasks |
| `ANTHROPIC_API_KEY` | *(unset)* | Required when backend=anthropic |
| `RESUME_PDFLATEX_CMD` | `pdflatex` | Set to `tectonic` if installed instead |
| `RESUME_MAX_UPLOAD_SIZE_MB` | `5` | Per-file cap |

## Why this design choice

- **Why HF as the default backend?** Free tier works for low-volume; resumes are short documents (a dozen LLM calls, total). Llama-3.1-8B is strong enough for editing tasks at this volume.
- **Why Claude as the swap-in?** Highest quality for instruction-following on editing tasks; the rules system prompt is reused across every section call, so prompt caching makes per-resume cost trivial.
- **Why a Jinja template, not in-place .tex editing?** In-place editing would have to mutate arbitrary user LaTeX — unsafe. Re-rendering from a canonical template is restrictive and predictable.
- **Why two safety guards?** LLMs occasionally over-summarize or drop a niche keyword. The guards make the "never degrade" promise enforceable, not just a policy hope.

## Limits

- The PDF parser uses heuristics for section detection. Highly stylized resumes may need a quick `.tex` conversion first (Overleaf does this).
- The compiled output uses a single canonical template (Jake-style). If your input had heavy custom styling, the enhanced `.tex` will look "cleaner" rather than "identical." This is intentional — the goal is a polished senior-grade output, not pixel-perfect round-tripping.
- The LLM is not asked to add new content (no fabricated metrics) — so the upgrade is a "lift", not a "rewrite". Resumes that are already strong will see small changes; weak resumes will see larger lifts.
