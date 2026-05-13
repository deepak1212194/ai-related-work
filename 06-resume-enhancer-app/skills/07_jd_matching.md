# JD Matching Skill — score the resume against a real-world job description

This step is deterministic — it does NOT call the LLM. Documented here so the
behaviour is auditable.

## Two evaluation paths

### Generic mode (curated JD templates)

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

When active, this produces a `JDMatchReport` stored in `PipelineResult.jd_report`.

### Custom mode (user-pasted JD)

When the user pastes a raw job description into the "Paste job description"
input, `JDMatchAgent.evaluate_custom()` is called instead of (or in addition to)
the generic path.

Keyword extraction from the raw text (`extract_keywords_from_jd()`):

1. Multi-word tech phrases are matched first via a regex bigram list
   (e.g. "machine learning", "fine-tuning", "multi-agent", "vector database").
2. Capitalised acronyms 2–8 chars long are extracted as must-haves
   (e.g. RAG, LLM, FAISS, CI/CD), excluding common English stop-words.
3. CamelCase library/framework names are extracted (e.g. PyTorch, LangChain).
4. Versioned tokens are extracted (e.g. "Python 3.x").
5. Section context determines required vs preferred: lines after "required",
   "must have", or "you will" map keywords to `must_have`; lines after
   "preferred", "nice to have", or "bonus" map to `nice_to_have`.
6. Generic noise words (deep, strong, years, familiar, etc.) are filtered out.

The extracted keywords are scored against the resume exactly as in generic mode
(must_have weighted 2x). Result is stored in `PipelineResult.custom_jd_report`.

**Enhancement integration:** when custom JD text is present, the extracted
`must_have` keywords are also merged into `priority_keywords` before the enhance
loop starts, so the AI actively targets that specific job's requirements rather
than generic role templates.

## gap_classification

After ATS scoring, missing keywords are split into two categories:

- **Presentation gaps** — keyword is present somewhere in the full resume IR
  text but was not picked up by the ATS surface scan. The user just needs to
  make it more visible (e.g. move it earlier, add it to the skills section).
- **Real gaps** — keyword is absent from the entire resume. Needs actual work:
  a project, a certification, or added experience before it can honestly appear.

Both lists are exposed in `ATSReport.presentation_gaps` and `ATSReport.real_gaps`
and shown in the Action Plan tab.

## fairness_note

This is keyword overlap, not semantic match. A JD asking for "vector search"
and a resume saying "FAISS-backed semantic retrieval" should still match —
that's why every JD entry has a `synonyms` field listing acceptable variants.
The custom JD path does not have a synonyms file; it relies on surface match
only, so the score may undercount semantic coverage.
