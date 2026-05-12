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
6. **Minimal patching.** Return only fields that need repair. Do not re-emit
   unchanged data.
7. **Links (LinkedIn / GitHub).** If `header.links` is missing a LinkedIn or
   GitHub entry but the raw .tex contains `linkedin.com/...` or `github.com/...`
   (inside `\href`, a bare URL, or a social command), add those links to
   `link_repairs` as `{kind, label, url}`. `kind` must be one of:
   `linkedin`, `github`, `email`, `website`, `twitter`, `scholar`, `portfolio`, `other`.
8. **Certifications.** If a `\section{Certifications}` or `\section{Licenses}` or
   similar block appears in the raw .tex but `certifications` is empty, extract
   each item as `{name, issuer, year}` into `certification_repairs`.
   Year should be a 4-digit string if present, otherwise empty string.

## schema_hints

- Skill items are short (≤ 4 words each), comma-separated within a bucket.
- Experience bullets are full sentences, one verb each.
- An experience `summary_line` is the optional one-line context after the
  company name (e.g. "OneForma AI Platform - Global Crowd-Work Marketplace").
- `link_repairs` items: `{"kind": "linkedin", "label": "LinkedIn", "url": "https://linkedin.com/in/..."}`
- `certification_repairs` items: `{"name": "AWS Solutions Architect", "issuer": "Amazon", "year": "2023"}`
