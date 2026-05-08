"""
Smoke test - parse the sample .tex, run the deterministic stages
(Completer + Planner default + Render), and write the output to .work/.

No LLM is called here, so no API key required.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.completer import CompleterAgent
from app.agents.jd_matcher import JDMatchAgent
from app.agents.planner import _default_plan, apply_plan
from app.core.ats import score_keywords
from app.core.config import workdir
from app.core.skills import get_bundle, load_skills
from app.parser import parse_tex_to_ir
from app.pipeline import list_role_keywords, ROLES
from app.render import render_ir_to_tex


logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    sample = ROOT / "examples" / "sample_input.tex"
    if not sample.exists():
        print(f"sample input missing: {sample}", file=sys.stderr)
        return 1

    log.info("loading skills...")
    bundle = load_skills()
    if not bundle.files:
        log.warning("no skill files loaded")
    log.info("parsing %s", sample)
    ir = parse_tex_to_ir(sample)
    log.info("name: %r", ir.header.name)
    log.info("links: %d", len(ir.header.links))
    log.info("summary: %r", ir.summary[:80])
    log.info("skills buckets: %d", len(ir.skills))
    log.info("experience blocks: %d", len(ir.experience))
    for i, e in enumerate(ir.experience):
        log.info("  [%d] %s @ %s (%s) - %d bullets", i, e.title, e.company, e.dates, len(e.bullets))
    log.info("education blocks: %d", len(ir.education))
    log.info("achievements: %d", len(ir.achievements))

    # Skip the Extractor (LLM only). Run Completer + default plan.
    completer = CompleterAgent(llm=None, skills=bundle)  # type: ignore[arg-type]
    completer.fill(ir)
    if ir.completed_fields:
        log.info("completed fields: %s", ir.completed_fields)

    plan = _default_plan(ir, "ai_ml_engineer")
    apply_plan(ir, plan)

    # Render
    tex = render_ir_to_tex(ir)
    out = workdir() / "smoke" / "resume.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    log.info("wrote %d bytes -> %s", len(tex), out)

    # Deterministic JD match
    text = ir.text_blob()
    role_keywords = list_role_keywords("ai_ml_engineer")
    ats = score_keywords(text, role_keywords)
    log.info("ATS (ai_ml_engineer): %.1f / %d keywords (matched %d)",
             ats.score, ats.total_checked, ats.matched_count)
    log.info("matched sample: %s", ats.matched[:8])
    log.info("missing sample: %s", ats.missing[:8])

    jd = JDMatchAgent()
    report = jd.evaluate(role_id="ai_ml_engineer", text_before="", text_after=text)
    log.info("JD samples: %d  avg_after=%.1f  top_gaps=%s",
             report.samples_count, report.avg_score_after, report.top_gaps[:6])

    print("\nOK — smoke parse + render + JD match succeeded.")
    print(f"Inspect: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
