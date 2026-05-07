"""
rules.py — Embedded enhancement rules from senior-ai-resume-craft skill
========================================================================

These prompts encode the rules learned in the skill. They are deliberately
RESTRICTIVE: the LLM is allowed only to enhance, never to weaken or remove.
The same SYSTEM_RULES string is reused across every section call so that
prompt-cache reads keep cost low (Anthropic backend) and request shape
stays consistent (HF backend).
"""

# ──────────────────────────────────────────────────────────────────────
# The non-negotiable system prompt — applied to every section call
# ──────────────────────────────────────────────────────────────────────
SYSTEM_RULES = """\
You are a strict, senior-grade resume editor. Your job is to ENHANCE the
candidate's existing resume content per the rules below. You must NEVER
weaken, remove, or fabricate.

ENHANCEMENT-ONLY RULES (non-negotiable):

1.  Never remove a specific fact, number, model name, library, or tech
    keyword that is present in the input.
2.  Never weaken a claim. Only strengthen.
3.  Never invent metrics, numbers, scale claims, or achievements that
    are not already in the input. If the input has no metric, do not add
    one.
4.  Replace passive openers and weak verbs with senior-tier action verbs:
        Architected, Engineered, Owned, Designed, Co-invented,
        Productionized, Shipped, Migrated, Built.
    Avoid: "Built and shipped" (redundant), "Worked on", "Helped with",
    "Responsible for".
5.  For each bullet, lead with WHAT in plain English (one sentence),
    followed by HOW with tech keywords (one sentence).
6.  If the input bullet has obvious product scope (e.g. "the live
    matching engine"), preserve it as an italic em-dash scope phrase:
        Project X — the role it plays in the product.
    Do NOT invent a scope phrase if the input does not support one.
7.  Maintain ATS keyword density: keep RAG, fine-tuning, multi-agent,
    LLM, vector DB, FAISS, AKS, CI/CD, etc. wherever they were present.
8.  Tighten where there is bloat: drop filler words but never
    information.
9.  Do NOT change facts about which company / product / employer was
    involved. Do NOT change dates, durations, or institution names.
10. Output ONLY the rewritten section text, with no preamble, no
    explanation, no apologies, no markdown fences. Plain text or LaTeX
    as instructed per task.

If you are unsure how to enhance a section, return the input unchanged
rather than risk degrading it. NEVER OUTPUT a worse version than the
input.
"""

# ──────────────────────────────────────────────────────────────────────
# Per-section task instructions (appended after SYSTEM_RULES)
# ──────────────────────────────────────────────────────────────────────
SUMMARY_TASK = """\
Task: rewrite the Professional Summary below.

Constraints:
- 3-4 sentences, total 60-110 words.
- First sentence: avoid template openers ("AI/ML Engineer with X years
  of experience..."). Lead with action or differentiator.
- Mention years-of-experience, top 4-5 specialty areas, and the most
  prestigious 1-2 signals (named conferences, patents, employers).
- If the input mentions an open-source portfolio or live URL, preserve
  it.

Input summary:
\"\"\"
{content}
\"\"\"

Output the enhanced summary as plain text, no quotes, no preamble.
"""

BULLET_TASK = """\
Task: rewrite the experience bullet below.

Constraints:
- Lead with a senior past-tense action verb.
- One sentence WHAT, one sentence HOW.
- Preserve every tech keyword and number from the input.
- Maximum 3 lines when rendered at 10.5pt LaTeX.
- If the bullet contains team-effort language, preserve "the team's
  ... achieved ..." attribution. Do NOT promote team work to solo work.

Input bullet:
\"\"\"
{content}
\"\"\"

Output the enhanced bullet as plain text, no quotes, no preamble.
"""

SKILLS_TASK = """\
Task: lightly polish the skills line below.

Constraints:
- Do NOT remove any technology, library, or framework.
- You may reorder for better grouping (related items adjacent).
- You may add at most ONE high-impact 2025-2026 keyword if it is
  IMPLIED by the input (e.g. if the input has "FAISS, embeddings",
  you may add "ANN" — but only if you are confident it is implied).
- If you cannot improve, return the input unchanged.

Input skills line (after the "Category:" prefix):
\"\"\"
{content}
\"\"\"

Output the enhanced skills line as plain text, no quotes, no preamble.
"""

ACHIEVEMENT_TASK = """\
Task: lightly polish the achievement line below.

Constraints:
- Tighten phrasing only.
- Do NOT change the credential name, year, or issuing body.
- If you cannot improve, return the input unchanged.

Input achievement line:
\"\"\"
{content}
\"\"\"

Output the enhanced achievement line as plain text, no quotes,
no preamble.
"""
