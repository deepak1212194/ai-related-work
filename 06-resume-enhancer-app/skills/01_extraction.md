# Extraction Skill — recover every field from a raw .tex file

You are reviewing a parsed resume that came out of a heuristic .tex parser. The
parser is imperfect: some fields will be missing, some will be jumbled, some
will be split across lines. Your job is to verify and complete the parse so the
downstream renderer has everything it needs.

## extraction_rules

1. **Do not invent.** If a field is missing from the input AND not derivable
   from another field, return an empty string. The Completer step (a separate
   agent) is the only one allowed to fill placeholders.

2. **Repair common parse errors.** If a degree's institution is in the
   "location" field, swap them. If an experience block has empty `title` but
   a clear "Senior Engineer at Acme" line in `bullets[0]`, lift it.

3. **Detect missed sections.** Re-read the raw text for any `\section{...}`
   we didn't classify. Common misses: Volunteering, Languages, Talks,
   Affiliations, Hackathons. These go into `extras`.

4. **Headline.** If `header.headline` is empty, derive a plausible one-line
   role descriptor from the most recent experience block — but ONLY using
   tokens that already appear in the input.

5. **Output format.** Respond with a JSON object matching the ResumeIR schema.
   No prose, no markdown.

## schema_hints

- Skill items are short (≤ 4 words each), comma-separated within a bucket.
- Experience bullets are full sentences, one verb each.
- An experience `summary_line` is the optional one-line context after the
  company name (e.g. "OneForma AI Platform - Global Crowd-Work Marketplace").
