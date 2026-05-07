# Skill: Section-Specific Task Prompts

> These are the per-section task instructions the agent uses when enhancing
> each resume section. Each task template has a `{content}` placeholder
> that gets filled with the actual section text.

## Summary Task

Rewrite the Professional Summary below.

Constraints:
- 3-4 sentences, total 60-110 words.
- First sentence: avoid template openers. Lead with action or differentiator.
- Mention years-of-experience, top specialty areas, and the strongest
  1-2 signals (named conferences, patents, employers).
- Preserve any open-source portfolio URL or live link from the input.

## Bullet Task

Rewrite the experience bullet below.

Constraints:
- Lead with a senior past-tense action verb.
- One sentence WHAT, one sentence HOW.
- Preserve every tech keyword and number from the input.
- Maximum 3 lines when rendered at 10.5pt LaTeX.
- If the bullet contains team-effort language, preserve "the team's
  ... achieved ..." attribution. Do NOT promote team work to solo work.

## Skills Task

Lightly polish the skills line below.

Constraints:
- Do NOT remove any technology, library, or framework.
- You may reorder for better grouping.
- You may add at most ONE high-impact 2025-2026 keyword if STRONGLY
  IMPLIED by the input (no fabrication).
- If you cannot improve, return the input unchanged.

## Achievement Task

Lightly polish the achievement line below.

Constraints:
- Tighten phrasing only.
- Do NOT change credential name, year, or issuing body.
- If you cannot improve, return the input unchanged.
