# Completion Skill — fill placeholders for missing required fields

The downstream renderer requires certain fields to produce a presentable resume:
name, headline, at least one contact link, summary, at least one skill bucket,
at least one experience entry, and at least one education entry.

When the input is missing one of these, you produce a clearly-marked placeholder
that the user can replace before sending the resume out. The placeholder must:

1. Use the exact tokens `[YOUR FULL NAME]`, `[your.email@domain.com]`,
   `[your-linkedin-handle]`, `[your-github-handle]`, `[CITY, COUNTRY]`,
   `[YEAR]`, `[INSTITUTION NAME]`, `[DEGREE]`, `[COMPANY NAME]`, `[ROLE TITLE]`,
   `[DATES]`, etc. — square brackets and ALL-CAPS so they are visually obvious.

2. Set `placeholder=true` on the relevant IR field so the UI can highlight it.

3. **Never invent a real-looking value.** Do not write a fake email, a fake
   employer, a fake degree. The placeholder must be obviously a placeholder.

4. For the summary, if missing, write a generic 1-2 line target-role-aware
   placeholder like:
   "[ADD 2-3 LINE SUMMARY: years of experience, primary domain, top
    technologies, one signature outcome]."

5. For an empty Experience section, emit ONE placeholder block with role title,
   company, dates, and three placeholder bullets, each starting with a senior
   action verb so the user has a working scaffold to fill in.

Output JSON only.
