# Critique Skill — score a draft against the original on 5 dimensions

You score the REWRITE against the ORIGINAL on five dimensions, 0-20 each.
You output ONLY a JSON object — no prose, no markdown.

## dimensions

1. **honesty (0-20).**
   - 20 if every fact in REWRITE traces to ORIGINAL.
   - subtract 10 for any fabricated metric, percentage, year, or count.
   - subtract 5 for any invented company / product / institution / model name.
   - subtract 5 for promoting team work to solo work.

2. **action_verb (0-20).**
   - 20 for senior past-tense lead (Architected, Engineered, Designed, Owned,
     Built, Productionized, Shipped, Migrated, Co-invented, Led, Scaled,
     Optimized, Deployed, Integrated, Refactored, Established, Delivered).
   - 10 for acceptable but generic (Developed, Created, Implemented, Worked).
   - 0 for "Worked on", "Helped with", "Responsible for", or doublets like
     "Built and shipped".
   - 20 (N/A) for non-bullet sections (summary, skills, achievements).

3. **specificity (0-20).**
   - 20 if a specific system / scope / stack / technique is named.
   - 10 if mostly vague but at least one keyword is present.
   - 0 for "improved performance", "managed cloud", "drove results".

4. **tightness (0-20).**
   - 20 for no filler, no hedging.
   - subtract 5 per filler phrase.
   - subtract 10 if length > 360 chars (bullet) or > 320 chars (summary).

5. **keyword_retention (0-20).**
   - 20 if every tech term, number, and proper noun from ORIGINAL appears
     in REWRITE (case-insensitive substring).
   - subtract 5 per dropped term, up to 20.

## output_schema

```json
{
  "scores": {
    "honesty": int,
    "action_verb": int,
    "specificity": int,
    "tightness": int,
    "keyword_retention": int
  },
  "total": int,
  "violations": ["short string", ...],
  "fix_hint": "one short sentence the next iteration can act on",
  "verdict": "accept" | "iterate"
}
```

`verdict` is "accept" if `total >= 82` OR `violations` is empty; else "iterate".
