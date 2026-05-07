# 06 · Resume Enhancer — Skill-Driven AI Agent

A **skill-file-driven resume enhancement agent** that reads its operating instructions from editable markdown files (`skills/*.md`), similar to how Claude uses skill files for context injection. Upload a PDF or `.tex` resume, pick a target role, and get a professionally polished `.tex` + `.pdf` with ATS keyword scoring.

## What it demonstrates

- **Skill file architecture**: All enhancement rules come from `.md` files loaded at runtime — edit them to change agent behavior without any code changes
- **AI agent with configurable instructions**: The LLM agent reads skill files for its system prompt, role emphasis, and per-section task templates
- **Hot-reload**: POST `/api/skills/reload` to pick up skill file edits without restart
- **ATS keyword scoring**: Automated Applicant Tracking System compatibility analysis
- **Safety guards**: Protected term detection, length validation, timeout caps — the agent never weakens or fabricates
- **Multi-backend**: Hugging Face (default) or Anthropic Claude with prompt caching

## Architecture

```
skills/
  core_rules.md          ──→  Base system prompt (non-negotiables)
  role_ai_ml_engineer.md ──→  Role-specific emphasis & keywords
  role_software_engineer.md
  role_data_scientist.md
  role_product_manager.md
  role_devops_cloud_engineer.md
  section_tasks.md       ──→  Per-section task templates

                 ┌──────────────────┐
  Upload ──→     │  Skill Loader    │ ──→ reads skills/*.md
  + Role         │  (hot-reloadable)│
                 └──────┬───────────┘
                        ↓
                 ┌──────────────────┐
                 │  LLM Agent       │ ──→ HuggingFace or Claude
                 │  (per-section)   │
                 └──────┬───────────┘
                        ↓
                 ┌──────────────────┐
                 │  Safety Guards   │ ──→ keyword check, length check
                 │  + ATS Scorer    │
                 └──────┬───────────┘
                        ↓
                    .tex + .pdf + ATS score
```

## Quick start

```bash
cd 06-resume-enhancer-app
pip install -r requirements.txt

# Default (HuggingFace backend)
export HF_API_KEY=hf_...
uvicorn app.main:app --reload --port 8006

# Or use Claude
export RESUME_LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-...
uvicorn app.main:app --reload --port 8006

# Open http://localhost:8006
```

## Customizing the Agent

Edit any file in `skills/` to change how the agent enhances resumes:

```bash
# Change core enhancement rules
vim skills/core_rules.md

# Add a new role profile
cp skills/role_ai_ml_engineer.md skills/role_blockchain_engineer.md
vim skills/role_blockchain_engineer.md

# Hot-reload without restart
curl -X POST http://localhost:8006/api/skills/reload
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status + backend + skill files loaded |
| `GET` | `/api/roles` | Available role profiles |
| `GET` | `/api/skills` | Skill file loading status |
| `POST` | `/api/skills/reload` | Hot-reload skill files |
| `POST` | `/api/enhance` | Upload + enhance → .tex/.pdf + ATS score |
| `POST` | `/api/ats-score` | Standalone ATS keyword analysis |
| `GET` | `/api/result/{id}.tex` | Download enhanced .tex |
| `GET` | `/api/result/{id}.pdf` | Download compiled .pdf |

## Safety guarantees

1. **Never weakens**: Output is always >= input quality
2. **Never fabricates**: No invented metrics, numbers, or achievements
3. **Protected terms**: Tech keywords, numbers, model names are never dropped
4. **Length guard**: Output must be >= 50% of input length (rejects truncation)
5. **Timeout cap**: Overall job timeout prevents runaway loops
6. **Structured errors**: Always returns a clean JSON response, never a stack trace

## Stack

FastAPI · Pydantic · Jinja2 · PyMuPDF · HuggingFace / Anthropic · pdflatex · Docker
