# Planning Skill — decide the section order, emphasis, and skill ordering

You are given an extracted ResumeIR and a target role. Decide:

1. **Section order.** The default is:
   `summary, skills, experience, projects, education, certifications,
   achievements, publications, extras`.
   Adjust per role:
   - Early-career or research-heavy candidates may benefit from
     `education -> experience` swap.
   - Engineering roles always lead with `skills` after summary.
   - Product roles lead with `experience` straight after summary, demoting
     `skills` to after `projects`.
   - Patents / publications are surfaced in the top half if the role is
     research-flavoured (AI/ML, Data Science, Research).

2. **Skill bucket ordering.** Reorder existing buckets so the ones most
   relevant to the target role come first. Do NOT add new buckets; do NOT
   delete existing buckets; do NOT add or remove items inside buckets
   (Enhancer step handles that under guard).

3. **Experience block ordering.** Most-recent-first by default. If the most
   recent role is a stop-gap (≤ 6 months) and a more senior earlier role is
   more relevant to the target, you may demote the stop-gap by ONE position
   and note the rationale.

4. **Bullet emphasis hints.** For each experience block, mark up to 2 bullets
   per block as "lead bullets" (those the Enhancer should over-invest in).
   Mark by index, never by rewriting them.

Output JSON:
```
{
  "section_order": ["summary","skills","experience",...],
  "skill_order_indices": [2,0,1,3,...],
  "experience_order_indices": [0,1,2,...],
  "lead_bullet_hints": {"0": [0,2], "1": [1]}
}
```
