# JD Matching Skill — score the resume against a real-world job description

This step is deterministic — it does NOT call the LLM. Documented here so the
behaviour is auditable.

For each curated JD in `data/jds/<role>.json`:

1. Tokenise the JD's `must_have` and `nice_to_have` keywords.
2. Search the resume text for each keyword as a whole-word substring,
   case-insensitive.
3. Score = `matched / total * 100`. The `must_have` set is weighted 2x.
4. The `delta` between original and enhanced resume is the headline metric
   the user sees — it answers "did the enhancement actually move the needle
   on this real job posting?"
5. The `top_gaps` field is the union of missing `must_have` keywords across
   all JDs for the role, capped at 10.

## fairness_note

This is keyword overlap, not semantic match. A JD asking for "vector search"
and a resume saying "FAISS-backed semantic retrieval" should still match —
that's why every JD entry has a `synonyms` field listing acceptable variants.
