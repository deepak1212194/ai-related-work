# Skill: Section-Specific Task Prompts

> These are the per-section task instructions the agent uses when enhancing
> each resume section. Each task template has a `{content}` placeholder
> that gets filled with the actual section text.
>
> The agent's effective system prompt for any section call is:
>   `core_rules.md` + `role_<chosen>.md` + the relevant Task section below.
>
> Headings drive the loader. Each `## X Task` becomes the key `x` in the
> `section_tasks` dict (e.g. `## Summary Task` → `summary`).

---

## Summary Task

Rewrite the Professional Summary below.

### Constraints

- 3-4 sentences, total 60-110 words.
- First sentence: avoid template openers ("X-year experienced AI/ML
  Engineer..."). Lead with action or differentiator.
- Mention years-of-experience, top specialty areas, and the strongest
  1-2 signals (named conferences, patents, prestigious employers).
- Preserve any open-source portfolio URL or live link from the input.
- Do NOT invent: no metrics, percentages, or scale numbers that aren't
  already in the input.
- Match the role-specific summary guidance from the loaded role file.

### Iteration rubric (apply mentally; max 3 iterations)

- ≥ 8/10 on Differentiation (no template opener)
- ≥ 7/10 on First-impression scan (one marquee credential by sentence 2)
- 100% Honesty / verifiability (every claim traces to input)
- 100% Corporate-safety (no internal product names, no non-public
  user counts)
- ATS keyword retention: every domain keyword from input present

If after 3 iterations the candidate still scores below the input's
baseline +3, return the input unchanged.

### Output

Plain text. No quotes. No preamble. No markdown fences.

### Input

```
{content}
```

---

## Bullet Task

Rewrite the experience bullet below.

### Constraints

- Lead with a senior past-tense action verb (Architected / Engineered /
  Owned / Designed / Co-invented / Productionized / Shipped / Migrated /
  Built). Reject openers like "Worked on", "Helped with",
  "Responsible for", "Built and shipped" (redundant).
- Structure: ONE sentence WHAT in plain English; ONE sentence HOW with
  tech keywords; OPTIONAL one clause for scale / metric / outcome ONLY
  if the input had it.
- Preserve every tech keyword and every number from the input.
- Maximum 3 lines when rendered at 10.5pt LaTeX (≈ 360 chars).
- If the input has product scope (this is the live X, the daily Y, the
  recruiter-side Z), preserve it as an italic em-dash phrase:
  `Project Name — the role this system plays in the product. Verb-led ...`
- If the bullet contains team-effort language, preserve "the team's
  ... achieved ..." attribution. Do NOT promote team work to solo work.
- Match the role-specific bullet guidance from the loaded role file.

### Iteration rubric (apply mentally; max 3 iterations)

- 100% on Honesty (no fabricated metric)
- ≥ 9/10 on Tone & language (action verb leads; no jargon stacking)
- 100% ATS keyword retention from input
- ≥ 7/10 on Quantified depth IF input had numbers (preserve them)
- ≥ 8/10 on Tightness (no "Built and shipped", no "Worked on")

Adjustment hints across iterations:

- **Iteration 1**: structural rewrite — verb at start, scope phrase if
  supported, WHAT then HOW.
- **Iteration 2**: tighten — cut filler words, merge clauses.
- **Iteration 3**: keyword polish — confirm every input keyword is back
  in. If still below baseline, preserve the input.

### Output

Plain text. No quotes. No preamble. No markdown fences. No bullet
character — just the rewritten text.

### Input

```
{content}
```

---

## Skills Task

Lightly polish the skills line below.

### Constraints

- Do NOT remove any technology, library, framework, or methodology.
- You MAY reorder items for better grouping (related items adjacent).
- You MAY add at most ONE high-impact 2025-2026 keyword if STRONGLY
  IMPLIED by the input. No fabrication. Examples of allowed additions:
  - Input has `FAISS, embeddings` → may add `ANN`
  - Input has `LangChain, LLM` → may add `Prompt Engineering`
  - Input has `Kubernetes, Docker` → may NOT add `Service Mesh` (not
    implied)
- Do NOT bold individual items inside the line. The bucket prefix is
  already bold; per-item bolding looks shouty and inconsistent.
- For techniques (vs. tools), italic is acceptable (e.g.,
  *Fine-tuning*, *Model Evaluation*).
- If you cannot improve without violating the rules, return the input
  unchanged. This is the explicit safe default.

### Iteration rubric (max 1 iteration; this is a polish, not a rewrite)

- 100% keyword retention
- ≥ 5/10 on Structure (related items grouped)
- 0% fabrication (no tech the input didn't have, except the single
  allowed implied addition)

### Output

Plain text. No quotes. No preamble. The output replaces the items
portion AFTER the bucket prefix (e.g. for input "Languages: Python,
SQL", output just "Python, SQL, Bash" — the loader re-prefixes).

### Input (after the bucket prefix)

```
{content}
```

---

## Achievement Task

Lightly polish the achievement line below.

### Constraints

- Tighten phrasing only.
- Do NOT change credential name, year, issuing body, or institution.
- Do NOT add details that weren't in the input.
- If the input is already terse and clear, return it unchanged.

### Iteration rubric (max 1 iteration)

- 100% credential preservation
- ≥ 6/10 on Tightness

### Output

Plain text. No quotes. No preamble.

### Input

```
{content}
```

---

## Education Task

(Optional — only invoked when the agent is configured to enhance
the Education section. By default, Education is preserved verbatim
to avoid altering institution names or dates.)

Rewrite the Education entry below.

### Constraints

- Do NOT change the degree name, institution, or dates.
- Do NOT add a percentage or CGPA. Senior 4+ year resumes drop these.
- You MAY add a one-line `Relevant coursework: ...` italic note if
  the institution is famous for a different specialty (e.g., ISI
  Kolkata is known for statistics; for an AI candidate, add a
  coursework note that lists ML / DL / Statistical Learning / etc.).
- You MAY note dissertation title in italic if present in input.
- If the input is already complete, return it unchanged.

### Output

Plain text or simple LaTeX. No quotes. No preamble.

### Input

```
{content}
```

---

## ATS Score Task

(Used for the standalone `/api/ats-score` endpoint, not in the main
enhancement flow.)

Score the resume content against an ATS keyword target. The agent
should return:

1. A coverage score 0-100 (% of role-relevant keywords from
   `role_<id>.md → Priority Keywords` that appear in the resume text).
2. List of keywords PRESENT in the resume (case-insensitive).
3. List of keywords MISSING from the resume.
4. List of TOP 5 keywords MOST LIKELY to be needed for the role
   that are missing — these are the highest-leverage gaps to fill.

### Output

JSON-shaped:

```json
{
  "score": 78,
  "present": ["Python", "PyTorch", "RAG", "FAISS", ...],
  "missing": ["Hugging Face", "fine-tuning", ...],
  "top_gaps": ["fine-tuning", "Hugging Face", "ANN", "MLflow", "model evaluation"]
}
```

### Input

```
{content}
```

---

## Iteration Discipline Reminder (applies to every Task)

- **Maximum 3 iterations** per section. After the third, preserve the
  input verbatim and emit a `kept — could not improve` note.
- **Stop early** if the iteration delta is < +2 points on the rubric.
- **Hard fail-safes** (always preserve input):
  - Output shorter than 50% of input
  - Any protected term from input dropped
  - Any new metric / percentage / scale number that wasn't in input
  - Any change to a company / product / employer / date / institution
    name
- **The agent's job is to LIFT, never to FLATTEN.** When in doubt,
  preserve the input. A clean preserve with a note is better than a
  shaky rewrite.
