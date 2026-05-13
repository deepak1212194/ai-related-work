# Technical Approach — AI Resume Enhancer

---

## Executive Summary

The AI Resume Enhancer is a multi-agent pipeline that transforms any LaTeX resume into
an ATS-optimised, Overleaf-ready `.tex` file. The system parses the input into a
structured intermediate representation (IR), enhances each section through a bounded
Enhancer+Critic loop, renders back to LaTeX, and scores the output against both a
deterministic ATS keyword model and curated real-world job descriptions. Every
agentic rewrite is gated by a deterministic Python safety guard, so no LLM output
can drop a fact, lose a protected term, or invent a metric that was not in the
original resume.

The architecture deliberately separates concerns into two layers: a small set of
LLM-powered agents that handle tasks requiring language understanding (parsing,
enhancement, hiring-manager simulation), and a larger set of deterministic Python
components that handle correctness guarantees (safety guard, ATS scoring, JD keyword
matching, placeholder filling, gap classification, history tracking). This separation
means the quality ceiling is set by the LLM but the correctness floor is set by
deterministic code that cannot be hallucinated away.

**v3 additions** (on top of the original 8-stage pipeline):

- **Custom JD targeting** — user can paste any raw job description. Keywords are
  extracted heuristically and merged into the enhancer's priority list, biasing
  rewrites toward that specific job's requirements. A separate custom JD match score
  shows before/after alignment for the pasted JD.
- **Gap classification** — missing ATS keywords are split into presentation gaps
  (already in the resume IR but not surface-visible) vs real gaps (absent entirely),
  so users know which gaps are easy wins and which require actual work.
- **Manual action checklist** — residual critic violations that the AI could not
  auto-fix are surfaced as a ranked, section-labelled checklist so users have
  specific next steps.
- **Run history** — every completed run is persisted to `.work/history.json`.
  Keywords that have been missing across 2+ runs appear as a "Recurring gaps" banner
  at the top of the Summary tab.
- **Redesigned UI** — sidebar layout with persistent run history, redesigned Action
  Plan tab, JD Match tab showing custom JD result first when present.

The primary LLM backend is Groq's free inference API. Groq provides a reliable
OpenAI-compatible endpoint with 14,400 free requests per day and multiple fallback
models. The pipeline is backend-agnostic via a `LLMClient` protocol, so swapping
in Anthropic, Hugging Face, or any future provider requires only a new backend
class — no changes to any agent.

---

## Problem Statement

Resume enhancement sits at the intersection of four hard constraints that make it
genuinely difficult:

**Information preservation.** A resume is a factual document. Any enhancement that
drops an employer name, changes a number, or omits a technology does active harm to
the candidate. Most LLM-based resume tools fail here: they produce fluent output that
silently loses content.

**Quality and ATS alignment.** Modern hiring funnels run every resume through an
applicant tracking system before a human sees it. A resume that scores poorly on
role-relevant keywords is filtered out regardless of the actual quality of the
candidate's work. Enhancement must increase keyword density without fabricating
experience.

**Template-agnostic parsing.** LaTeX resumes are written in dozens of templates:
Jake's Resume, Awesome-CV, ModernCV, custom Overleaf forks, and fully hand-rolled
documents. A regex-only parser that knows about one template structure will fail on
all others. The parser must be robust enough to handle arbitrary command names
(`\documentTitle`, `\tinysection`, `\resumeSubheading`, etc.) while still producing
a uniform internal representation.

**Output fidelity.** The output must be a compilable LaTeX file the candidate can
open in Overleaf immediately. This rules out outputting Markdown, plain text, or PDF
and requires the render layer to produce correct LaTeX escaping, proper environments,
and a professionally typeset layout.

---

## Architecture Overview

```
  .tex  +  optional custom JD text
               |
               +--------------------------------------------------+
               |               run_pipeline()                     |
               |                                                  |
               | 1-Parse  2-Repair  3-Complete  4-Plan            |
               |                    |                             |
               |   custom JD keywords merged into priority_kw     |
               |                    |                             |
               |              5-Enhance loop                      |
               |         +----------+----------+                  |
               |         |                     |                  |
               |   EnhancerAgent         CriticAgent              |
               |   draft_block()         score_block()            |
               |         |                     |                  |
               |         +-----> safe_apply() -+                  |
               |                (deterministic)                   |
               |                    |                             |
               |  6-Render  7-Score  8-Review  9-History          |
               |                                                  |
               +--------------------------------------------------+
                                    |
               +--------------------+--------------------+--------+
               |                    |                    |        |
            .tex out           ATS report           Role review  History
           (Overleaf)     + gap classification    (JSON: score,  (.work/
                          + custom JD report       strengths,   history.json)
                          + manual actions         gaps)
```

The pipeline is synchronous and single-threaded per job. All I/O (file reads,
LLM calls, job directory writes) is explicit. The Gradio UI runs the pipeline
on a background thread and streams progress events to the frontend via a queue.

---

## Stage-by-Stage Explanation

### Stage 1: LLM-First Parse

**Why not regex-first?**

A regex-first parser must know the specific LaTeX macro names used by a template.
Jake's Resume uses `\resumeSubheading{company}{dates}{title}{location}` while
ModernCV uses `\cventry{dates}{title}{company}{location}{}{}`. Trying to enumerate
all templates produces a brittle mapping that breaks on any custom or hand-rolled
document.

The LLM-first approach sends the raw `.tex` content (up to 12,000 characters —
approximately 3,000 tokens, which fits comfortably within the 128K context window of
all supported Groq models) to the LLM with an explicit JSON schema. The LLM is
instructed to "find content regardless of the command names used" and to extract name,
headline, contact links, summary, skills, experience, projects, education,
certifications, and achievements. Because the LLM understands LaTeX syntax and
semantics rather than pattern-matching on specific command names, it handles
arbitrary templates correctly.

The regex parser (`app/parser/tex_parser.py`) is retained as a fallback. If the
LLM fails, returns non-JSON, or returns a response that fails the is-resume-like
check (requires at least 2 of: name, summary, skills, experience, education), the
regex parser runs instead. This dual-layer approach means the pipeline never fails
to produce a parse.

**12,000-character window.** Real resumes almost never exceed 12,000 characters.
The window covers any real-world resume while keeping the extraction prompt well
under the model's context limit.

### Stage 2: Repair

The repair stage runs only when the regex parser was used as the fallback — the LLM
extraction path is treated as a complete parse and does not need repair. When repair
does run, `ExtractorAgent.repair()` constructs a compact excerpt of the raw `.tex`
(prioritising `\begin{center}`, `\section`, and `\name{` regions) and asks the LLM
to return a minimal JSON patch with only the fields that need correction.

Repair targets five specific failure patterns:
- Missing `header.headline` — lifted from the most recent experience block
- Experience blocks with empty `title` — recovered from the first bullet if it
  matches an "Engineer at Company" pattern
- Education blocks with institution and location fields swapped
- LinkedIn and GitHub links present in the raw `.tex` but missed by the regex parser
- Certifications in an unparsed `\section{Certifications}` block

The repair is strictly additive. It cannot modify fields that already have values,
cannot reorder sections, and cannot add new experience blocks. This bounds the
blast radius of any LLM repair error.

### Stage 3: Complete

The `CompleterAgent` is entirely deterministic — it does not call the LLM. It
inspects the `ResumeIR` for required fields and inserts `[PLACEHOLDER]` tokens
(e.g., `[YOUR FULL NAME]`, `[your.email@domain.com]`, `[COMPANY NAME]`) wherever
a required field is missing. Placeholder tokens are ALL-CAPS inside square brackets
so they are visually obvious to the user and cannot be confused with real content.

The field `placeholder=True` is set on any IR object that contains a placeholder,
which allows the Gradio UI to highlight unfilled sections after the run completes.

The design choice to use explicit placeholder tokens rather than LLM-generated
plausible values is deliberate: guessing a plausible-looking company name or email
address would produce content that looks real but is fabricated. Placeholder tokens
make the gap visible and force the user to fill it rather than accidentally submitting
a resume with invented data.

### Stage 4: Plan

`PlannerAgent` decides three things for the downstream enhancement and render stages:
section ordering, skill bucket ordering, and lead-bullet hints.

**Section ordering** follows role-specific rules encoded in the planning skill file.
Engineering roles (AI/ML Engineer, Software Engineer, DevOps) surface `skills`
immediately after the summary. Product Manager roles demote `skills` to after
`projects` and lead with experience. Research-flavoured roles (AI/ML Engineer,
Data Scientist) surface patents and publications in the top half.

**Skill bucket ordering** reorders existing buckets (e.g., "ML Frameworks" before
"Languages" for an AI/ML Engineer target) without adding or removing any items —
item-level changes are left to the Enhancer under the safety guard.

**Lead-bullet hints** mark up to 2 bullets per experience block as high-priority.
The Enhancer receives a `LEAD_BULLET` signal for these bullets and is instructed to
invest more in scope phrase and specific measured outcome. This avoids flattening
all bullets to the same level of polish.

### Stage 5: Enhancement Loop

This is the core of the pipeline. The `IterativeOrchestrator` coordinates the
`EnhancerAgent` and `CriticAgent` in a bounded feedback loop.

**Block-level batching.** Rather than one LLM call per bullet (which would produce
60–120 calls for a typical 20–40 bullet resume at 3 iterations), the orchestrator
calls `EnhancerAgent.draft_block()` once per experience or project block. A block
with 5 bullets becomes 1 LLM call that receives all 5 bullets in `[N] text` format
and returns 5 rewrites in the same format. For a typical resume this reduces total
LLM calls by 3–6x.

**Output format.** The block prompt specifies the exact output contract: return
exactly N lines, each prefixed `[N]`, no markdown, no explanations. The
`_parse_block_response()` function in `enhancer.py` uses a simple regex
(`^\[(\d+)\]\s+(.+)`) to extract the indexed lines. If fewer than half the expected
bullets are parsed, the orchestrator falls back to per-bullet `draft()` calls,
so accuracy is never compromised by a partial parse.

**Iteration bounds.** The loop runs for at most `min(max_iterations, 4)` rounds.
The `RESUME_MAX_ITERATIONS` setting (default: 3) controls the typical cap; the hard
ceiling of 4 cannot be exceeded by configuration. Early acceptance fires when the
Critic returns `score >= RESUME_ACCEPT_THRESHOLD` (default: 82) or `verdict=accept`.
The `RESUME_MIN_DELTA` guard stops iterations when improvement between rounds is less
than 3 points, preventing wasted calls on a plateau.

**Critic scoring.** The Critic evaluates each rewrite on 5 dimensions (0–20 each,
total 0–100):
- `honesty` — fabricated metrics, invented names, promoted solo work
- `action_verb` — senior past-tense lead verb quality
- `specificity` — system/scope/stack/technique named
- `tightness` — filler, hedging, length violations
- `keyword_retention` — every protected term from original present in rewrite

The Critic returns structured JSON with `scores`, `total`, `violations`, `fix_hint`,
and `verdict`. The `fix_hint` is passed to the next Enhancer iteration so it can
address specific violations rather than generating a fresh draft blind.

**Safety guard.** After the loop completes, `safe_apply()` runs a deterministic
Python check on the winning draft:
1. The rewrite must be non-empty.
2. The rewrite length must be between 50% and 400% of the original length.
3. Every protected term from the original must be present in the rewrite
   (case-insensitive substring match).

If any check fails, the original text is kept and the reason is recorded in the
`SectionTrace.note` field. This is the final and authoritative decision — it
overrides the Critic's verdict.

### Stage 6: Render

`render_ir_to_tex()` passes the enhanced `ResumeIR` to a Jinja2 template
(`app/render/template.tex.j2`). The template is based on Jake's Resume (MIT licensed)
with professional typographic refinements: a two-column header with FontAwesome5
icons for contact links, colored section rules, and a 10.5pt Source Sans Pro body
font.

The output is a `.tex` string that can be compiled directly with `pdflatex` or
uploaded to Overleaf. The choice to output `.tex` rather than PDF is intentional:
`.tex` is editable and allows the candidate to make post-enhancement adjustments,
fill any remaining placeholder tokens, and customise typographic details. PDF output
would lock in the current state and require a full re-run for any modification.

The render stage does not call the LLM — it is purely a template fill. LaTeX special
characters (`&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`, `\`) are escaped in all
string fields before template substitution.

### Stage 7: ATS Scoring, JD Matching, Gap Classification, and Manual Actions

**ATS scoring** is deterministic keyword frequency analysis. The pipeline extracts
the role's `priority_keywords` list from the role skill file and counts how many
appear in the enhanced resume text (whole-word, case-insensitive). The result is
a score 0–100, a list of matched keywords, and a list of missing high-impact keywords.
No LLM is involved.

**Gap classification** (`_classify_gaps()` in `pipeline.py`) splits the missing
keywords into two categories:
- *Presentation gaps* — the keyword is present somewhere in the full `ResumeIR`
  text blob but was not picked up by the ATS surface scan. These are easy wins:
  the candidate already has the skill, they just need to make it more visible
  (move it to the skills section, add it to a bullet).
- *Real gaps* — the keyword is absent from the entire resume. These require actual
  work before the candidate can honestly claim them.

Both lists are exposed in `ATSReport.presentation_gaps` and `ATSReport.real_gaps`
and shown in the Action Plan tab.

**JD matching — generic path** compares the resume text against curated job description
keyword sets stored in `data/jds/<role_id>.json`. Each JD entry has `must_have`
keywords (weighted 2x) and `nice_to_have` keywords (weighted 1x). The `JDMatchAgent`
computes a per-JD match score before and after enhancement. The headline metric shown
to the user is `avg_delta`: the average improvement in JD match score across all
curated JDs for the target role. The `top_gaps` field lists the must-have keywords
still missing after enhancement.

JD keyword matching uses a `synonyms` field in each JD entry to handle semantic
variants. A JD asking for "vector search" matches a resume containing "FAISS-backed
semantic retrieval" if "FAISS" or "semantic retrieval" appears in the synonyms list.

**JD matching — custom path** (`JDMatchAgent.evaluate_custom()`) scores the resume
against keywords extracted from the user's pasted JD text. The extraction heuristic
in `extract_keywords_from_jd()`:
1. Matches multi-word tech phrases from a regex bigram list (fine-tuning,
   multi-agent, vector database, etc.).
2. Extracts capitalised acronyms (RAG, LLM, FAISS, CI/CD).
3. Extracts CamelCase library/framework names (PyTorch, LangChain).
4. Uses section context (lines after "required" / "preferred") to split results
   into `must_have` vs `nice_to_have`.
5. Filters generic noise words (deep, strong, years, familiar, etc.).

The extracted keywords are also merged into `priority_keywords` before the enhance
loop starts (Stage 5), so the AI actively targets the specific job's requirements.
The custom JD report is stored in `PipelineResult.custom_jd_report` and shown first
in the JD Match tab.

**Manual action checklist** (`_build_manual_actions()`) scans all `SectionTrace`
objects after the enhance loop. Any section that still has unresolved critic
violations and a final score below 80 becomes a `ManualActionItem` with the section
label, the top violation, and the critic's fix hint. Items are sorted by urgency
(lowest score first) and capped at 20. They appear in the Action Plan tab as an
explicit to-do list for the user.

### Stage 9: History

After every completed run, `app/core/history.py` appends a `RunRecord` to
`.work/history.json`. Each record stores role, ATS/HM/JD scores, elapsed time,
missing keywords, gap lists, and a flag for whether a custom JD was used. No PII
(no resume text, no personal details) is stored.

The persistent gap tracker maintains a counter for every missing keyword across runs.
Keywords that appear in 2 or more runs are surfaced as a "Recurring gaps" banner at
the top of the Summary tab, prompting the user to address them directly rather than
relying on the AI to work around them.

### Stage 8: Role Review

`RoleReviewerAgent` is the only stage that asks the LLM to produce a qualitative
judgment rather than a structured transformation. It receives the full enhanced resume
text and the role profile, and plays the role of a senior hiring manager at a
top-tier company evaluating the candidate for that specific role.

The review returns structured JSON: `overall_score` (0–100 probability of advancing
to phone screen), `strengths` (3–5 items), `weaknesses` (2–4 items),
`missing_keywords` (up to 10), and `one_line_verdict`. The skill file instructs the
reviewer to cap output to these bounds and to never recommend changes that would
require fabrication — if the resume genuinely lacks experience the role needs, it is
listed as a weakness, not as a bullet rewrite suggestion.

The LLM is used here (rather than a deterministic method) because the hiring-manager
simulation requires synthesising multiple signals — section quality, keyword
coverage, seniority calibration, coherence of the technical narrative — into a
holistic judgment. This is a task where LLM reasoning is genuinely superior to rule
enumeration.

---

## Agent Design Rationale

### Why Enhancer+Critic (Not a Larger Crew)

The alternative of assigning a specialised agent per section type (one for summaries,
one for bullets, one for skills, one for achievements) adds coordination overhead
without adding quality. Each specialised agent still needs the same core rules,
role context, and safety guard. The Enhancer already handles all section types via a
`ACTIVE_SECTION_TYPE` parameter in its system prompt, which selects the appropriate
rules from `skills/04_enhancement.md`.

The Critic is valuable precisely because it provides a single consistent scoring
rubric across all section types. A per-section critic would require each to learn
the same rubric independently, introducing variance. The 5-dimension JSON schema
is simple enough that even an 8B parameter model reliably produces parseable output.

### Why Deterministic Safety Guard (Not Trust the LLM)

LLMs are probabilistic. Even with explicit instructions not to drop terms, a model
under token pressure may silently omit "FAISS" or change "2.3M users" to "millions
of users". The cost of such an omission is real: the candidate submits a resume with
falsified content. The deterministic `safe_apply()` guard eliminates this failure
mode at zero token cost and with zero latency. Trust the LLM for quality improvement;
trust the deterministic guard for correctness.

The guard also functions as a circuit breaker. When the LLM is having a bad day
(returning short, repetitive, or otherwise degenerate output), the guard catches
the regression and silently keeps the original. The candidate never receives a
degraded version.

### Why Hot-Reloadable Skill Files (Not Hardcoded Prompts)

Hardcoding instructions in Python string literals makes iterative improvement
expensive: every change requires a code edit, commit, and restart. Skill files in
`skills/*.md` can be edited while the service is running; the next pipeline run calls
`load_skills()` at the start of `run_pipeline()` and picks up the changes immediately.

This design also enables non-engineers to tune agent behaviour. A domain expert can
add calibration examples to `skills/04_enhancement.md` or adjust the critic rubric
in `skills/05_critique.md` without touching any Python. The Docker compose
configuration mounts `skills/` as a read-only volume for exactly this reason.

The skill loader parses named `## block` sections within each markdown file. Agents
can request specific blocks (e.g., `priority_keywords` and `hiring_signals` from a
role file) rather than loading the full file, keeping system prompts compact and
within the token budget.

### Why Block-Level Batching (3–6x Fewer LLM Calls)

Per-bullet LLM calls treat each bullet as an independent problem, discarding the
contextual relationship between bullets in the same experience block. A batched block
call gives the model the full block context (`BLOCK_CONTEXT: CompanyName — Title
(dates)`) so it can maintain consistent tone, avoid repetitive action verbs, and
understand cross-bullet dependencies (e.g., if bullet 1 introduces a system, bullet 2
can reference it by name).

The `[N]` indexed output format is compact (no JSON overhead, no prose wrapper) and
tolerant to partial failure: if 4 of 5 bullets parse correctly, the 5th slot falls
back to the original rather than failing the whole block. The threshold for triggering
a per-bullet fallback is when fewer than `max(1, n//2)` bullets parse successfully.

Token budgets scale with block size: `max_tokens = min(130 * n, 1600)`, capping at
1,600 output tokens per block call to prevent runaway generation while giving enough
room for verbose bullets.

---

## Information Preservation Philosophy

### Protected Terms System (Dynamic + Baseline)

Protected terms come from two sources that are unioned at runtime:

**Baseline set** (`safety.py`): approximately 100 terms covering common ML/cloud/dev
tools that appear in almost every AI engineer resume — PyTorch, TensorFlow, FAISS,
BERT, Azure, Kubernetes, Docker, etc. The baseline ensures that even a resume with a
very sparse skills section has its technology stack protected.

**Dynamic set** (`extract_protected_terms_from_ir()`): extracted from the parsed
`ResumeIR` before the enhancement loop starts. Every item in every skill bucket,
every company and job title, every project name and stack item, every institution
name, every certification name, and every publication title is added to the dynamic
set. This guarantees that user-specific terms not in the baseline (e.g., an internal
tool name, a niche framework, or a company name) are protected.

The union is passed to the `IterativeOrchestrator` at construction time and applied
in every `safe_apply()` call throughout the enhancement loop.

### Numbers Policy

Numbers in resumes — percentages, latency figures, user counts, team sizes — are
the hardest facts to protect and the easiest to distort. The `00_core_rules.md`
skill file specifies an explicit numbers policy:
- If the input names a number, keep it verbatim.
- If the input gives a fuzzy descriptor ("tens of thousands"), keep the same
  descriptor — do not sharpen it to an exact count.
- Round only when the input itself rounds. Never invent a rounded estimate.

The `_PROTECTED_PATTERNS` list in `safety.py` encodes number patterns as regex:
years (`\b(19|20)\d{2}\b`), percentages (`\b\d+(?:\.\d+)?\s*%`), multiples
(`\b\d+(?:\.\d+)?x\b`), counts with units (M, K, B, GB, MB, ms, GPUs), and currency
amounts. Any number matching these patterns in the original is added to the protected
set and must appear verbatim in the rewrite.

### Years-of-Experience Protection

Year strings in dates ranges (e.g., "2021–2024", "Jan 2022 – Present") are matched
by the `\b(19|20)\d{2}\b` pattern and protected. The enhancer cannot change "4 years
of experience" to "5+ years" and cannot extend a date range. This prevents the
LLM from accidentally seniority-inflating a candidate's tenure.

### Guard Hierarchy

The full decision hierarchy, from highest to lowest priority, is:

1. **Deterministic safety guard** (`safe_apply()`) — final word; cannot be overridden.
2. **Critic verdict** — informs the accept/iterate decision inside the loop.
3. **Critic score** — determines early acceptance at threshold.
4. **Enhancer output** — the actual proposed rewrite.
5. **Skill file instructions** — guide the enhancer's style.

The guard sits above the critic in the hierarchy. An LLM that returns `verdict=accept`
with `score=100` but drops a protected term still loses to the guard, which restores
the original.

---

## LLM Backend Design

### Why Groq Primary (Not HuggingFace by Default)

Groq provides a reliable, fast, OpenAI-compatible inference endpoint on a generous
free tier (14,400 requests per day as of May 2026). HuggingFace Inference API is
free but model availability is intermittent — models go offline, responses are slow
during peak demand, and the serverless router sometimes returns 404 for a model that
was working an hour earlier. Groq's reliability and speed make it the better default
for a production-grade application. HuggingFace is retained as a secondary backend for
the extraction stage when `RESUME_ENABLE_MULTI_LLM=true`, allowing a cheaper model
to handle parsing while Groq handles the quality-sensitive enhancement loop.

### Auto-Fallback Chain (Model Rotation)

The `GroqLLM` class maintains an ordered fallback list:

1. `meta-llama/llama-4-scout-17b-16e-instruct` — 30,000 TPM (best headroom)
2. `llama-3.3-70b-versatile` — 6,000 TPM (highest quality)
3. `llama-3.1-8b-instant` — 6,000 TPM (fastest)

When the active model returns HTTP 429 (rate limit) or a "decommissioned" error, the
client rotates to the next model in the list. Each model has its own independent TPM
bucket, so a 429 on `llama-3.3-70b-versatile` does not affect `llama-4-scout`. The
rotation is transparent to all agents — they call `llm.complete()` and receive text;
model selection is entirely encapsulated in the backend.

The `HuggingFaceLLM` class maintains a similar fallback list of open-access models:
`mistralai/Mistral-7B-Instruct-v0.3`, `HuggingFaceH4/zephyr-7b-beta`,
`Qwen/Qwen2.5-7B-Instruct`, `microsoft/Phi-3.5-mini-instruct`, and
`google/gemma-2-2b-it`. It also tries four endpoint variants per model (HF router,
inference API, model-specific chat, text-generation) to maximise the chance of a
successful call.

### Retry Logic (Parse retry-after from Response)

The `_retry_call()` wrapper applies to all backends. For HTTP 429 responses, it
parses Groq's embedded retry-after message: `"Please try again in X.Xs"`. If the
message is present, the wrapper waits exactly that duration plus a 1.5-second buffer.
If the message is absent, it defaults to a 12-second wait with 0–2 seconds of jitter.
For other transient failures (timeouts, server errors), exponential backoff with jitter
applies: `delay = base_delay * 2^attempt + uniform(0, 0.5)`. Auth errors
(`401`, `403`) are never retried — they indicate a permanent configuration problem.

---

## Skill File Architecture

### What Skill Files Contain

Each `skills/*.md` file encodes one agent's operating instructions. The file has a
free-form `# Title` line (ignored by the loader), an optional default block (content
before the first `## heading`), and named blocks (each starting with `## block_name`).

Named blocks allow agents to request only the high-signal sections of a file rather
than loading all content. For example, the `EnhancerAgent` requests only
`priority_keywords`, `hiring_signals`, and `red_flags` from the active role file
(approximately 200 tokens), rather than the full role file which includes
`seniority_calibration` and other blocks not needed by the enhancer.

Role files (`role_*.md`) have a standardised block structure:
- `## priority_keywords` — comma-separated list of role-relevant terms
- `## hiring_signals` — what a hiring manager looks for (positive signals)
- `## red_flags` — what a hiring manager penalises (negative signals)
- `## seniority_calibration` — IC4/IC5/IC6/IC7 expectations

### Hot Reload Mechanism

`skills.py` maintains a module-level `_BUNDLE` variable protected by a threading
lock. `load_skills()` re-reads all `*.md` files from the `skills/` directory and
atomically replaces `_BUNDLE`. `run_pipeline()` calls `load_skills()` at the start
of every job, so edits made to skill files between runs are always picked up.

In the Docker deployment, `skills/` is mounted as a read-only volume. An operator
can `scp` or edit a skill file and the next pipeline run will use it without any
container restart or image rebuild.

### How Agents Consume Skill Files

Every agent receives a `SkillBundle` via dependency injection in its `__init__`. The
bundle exposes:
- `get_block(file_name, block_name)` — get a specific named block from a skill file
- `get_role(role_id)` — get the full role profile for a given role
- `get_role_blocks(role_id, block_names)` — get only the specified blocks from a role
  file (used by `EnhancerAgent` to control prompt token size)

The `EnhancerAgent.system_prompt()` method assembles the system prompt by composing
`00_core_rules` (the hard rules), `04_enhancement` (the enhancement task), and the
role's `priority_keywords` + `hiring_signals` + `red_flags` blocks. The total system
prompt is typically 800–1,200 tokens.

---

## Extension Guide

### Adding a New Role

1. Create `skills/role_<role_id>.md` with the standard block structure
   (`## priority_keywords`, `## hiring_signals`, `## red_flags`,
   `## seniority_calibration`).
2. Add the role to the `ROLES` dict in `app/pipeline.py`:
   ```python
   ROLES["my_new_role"] = "My New Role Display Name"
   ```
3. Optionally add a JD keyword file at `data/jds/my_new_role.json` with the
   structure `[{"title": "...", "must_have": [...], "nice_to_have": [...],
   "synonyms": {...}}]`.
4. No code changes needed to any agent. The role skill file is discovered
   automatically by the skill loader.

### Adding a New Backend

1. Implement the `LLMClient` protocol (`app/core/llm.py`):
   - `name: str` attribute
   - `complete(system, user, *, max_tokens, temperature, cache_system) -> str`
   - `last_usage() -> LLMUsage`
2. Add a detection function `is_backend_configured("my_backend")`.
3. Add a branch in `build_llm()` to instantiate your class.
4. Add your backend to `best_available_backend()` in the priority order you want.

No changes to any agent class are required. The `LLMClient` protocol ensures
complete backend transparency.

### Changing Enhancement Rules

- **Add/remove diction blacklist terms:** edit the `## diction_blacklist` block in
  `skills/00_core_rules.md`. Changes take effect on the next pipeline run.
- **Add bullet calibration examples:** add a new `### bullet_examples` subsection in
  `skills/04_enhancement.md` with INPUT/OUTPUT pairs. More examples improve
  consistency with smaller models.
- **Change the critic rubric:** edit per-dimension scoring rules in
  `skills/05_critique.md`. Add calibration examples to the `## calibration_examples`
  section to guide the model's scoring.
- **Change accept threshold:** set `RESUME_ACCEPT_THRESHOLD` in `.env`. The range
  is 0–100; 75 is permissive, 90 is very strict.
- **Adjust length limits:** the bullet length cap (360 chars) and summary length cap
  (320 chars) are enforced both in the enhancement skill file instructions and in the
  Critic's tightness dimension. Changing one without the other creates inconsistency.

---

## Known Limitations and Future Work

**Template support.** The LLM-first extractor handles arbitrary templates well but
occasionally misparses complex nested LaTeX (e.g., multi-column `tabularx` layouts
or `tikz`-based designs). The fix is to add more template-specific heuristics to the
repair stage or expand the extraction prompt examples.

**Multi-page resumes.** The 12,000-character extraction window covers any standard
one-page resume and most two-page resumes. Very dense two-page or three-page resumes
may be truncated. Future work: chunk large inputs and merge the resulting IRs.

**Groq TPM limits on large resumes.** A resume with 40 bullets across 8 blocks at
3 iterations uses approximately 24 LLM calls (8 block enhance + 8 block critique,
×3 iterations, with early accepts reducing this). On `llama-3.1-8b-instant` at
6,000 TPM and ~200 tokens per call, a full run consumes roughly 5,000 tokens — close
to the per-minute limit. The auto-fallback to `meta-llama/llama-4-scout-17b-16e-instruct`
(30,000 TPM) handles this. Future work: add a `speed` mode that skips the critic for
already-strong bullets (the `_should_skip_rewrite()` heuristic is in place but not
fully wired to the critic path).

**JD data freshness.** The curated JD keyword sets in `data/jds/` reflect job
postings as of mid-2025. JD requirements evolve; a future maintenance task is to
refresh the keyword sets quarterly from fresh job postings.

**No PDF input support.** The pipeline accepts `.tex` only. Supporting PDF input
would require a PDF-to-text extraction step (PyMuPDF is already in the dependencies)
followed by wrapping the extracted text in a minimal LaTeX skeleton before the normal
parse path. This is a planned enhancement.

**Cross-role scoring is opt-in.** The `enable_cross_role` flag in `PipelineConfig`
enables scoring against all 5 supported roles. It is off by default because it adds
5 additional role review calls and 5 additional JD match evaluations per run. A
future UI improvement would show a cross-role radar chart without requiring full
cross-role processing.

**Custom JD scoring is surface-level only.** The custom JD path uses heuristic
keyword extraction and surface-match scoring. Unlike the curated JD path, it has no
`synonyms` file, so semantic variants are not recognised (e.g. "FAISS" in the resume
will not match "vector search" in the custom JD). As a practical workaround, paste JDs
with explicit technology names rather than abstract descriptions.

**Custom JD keyword extraction is heuristic.** The bigram list and noise filter cover
most standard tech JDs, but highly domain-specific terminology (e.g. proprietary
platform names, niche scientific toolkits) may not be captured. Power users can extend
`_TECH_BIGRAMS` in `jd_matcher.py` to add domain-specific phrases.

**History is local and instance-scoped.** The run history in `.work/history.json` is
per-deployment. It is not shared across Docker instances or reset-proof unless `.work/`
is mounted as a persistent volume. The persistent-gaps callout only reflects runs
within the same history file. To reset: `python -c "from app.core.history import
clear_history; clear_history()"`.

**Manual actions cap at 20.** The checklist is truncated to the 20 lowest-scoring
residual violations. On resumes with many sections, lower-urgency issues above the cap
are still visible in the Sections tab's per-section critic detail.
