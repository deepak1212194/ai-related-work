"""
gradio_app.py - polished, multi-tab Gradio UI for the Resume Enhancer.

Tabs:
  - Enhance       upload .tex, pick role, run pipeline.
  - Sections      per-section before/after with critic trace.
  - Review        single-focus hiring-manager simulation for the target role.
  - JD Matching   keyword score against curated JDs for the target role.
  - Download      .tex output + Overleaf paste-ready helper.
  - Setup         backend status + step-by-step auth options.
  - About         architecture, agents, tech stack.

Run with:
    python -m ui.gradio_app
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings   # noqa: E402
from app.core.ir import PipelineResult  # noqa: E402
from app.core.llm import (best_available_backend, detect_available_backends,  # noqa: E402
                          is_backend_configured)
from app.core.skills import load_skills  # noqa: E402
from app.pipeline import (PipelineConfig, ROLES, list_role_keywords,  # noqa: E402
                          run_pipeline)


logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)

load_skills()


# ----------------------------------------------------------------------
# CSS - designer-tier: refined typography, motion, depth, color system.
# ----------------------------------------------------------------------
CSS = """
/* ---------------- Design tokens ---------------- */
:root {
  --rx-bg-page:   #f6f7fb;
  --rx-bg-0:      #ffffff;
  --rx-bg-1:      #fafbfd;
  --rx-bg-2:      #f3f4f8;
  --rx-line:      #e7e9ee;
  --rx-line-2:    #d6d9e0;
  --rx-ink:       #0f172a;
  --rx-ink-2:     #475569;
  --rx-ink-3:     #6b7280;
  --rx-ink-4:     #94a3b8;
  --rx-brand:     #4f46e5;
  --rx-brand-2:   #6366f1;
  --rx-brand-50:  #eef2ff;
  --rx-mint:      #10b981;
  --rx-amber:     #f59e0b;
  --rx-rose:      #ef4444;
  --rx-violet:    #8b5cf6;
  --rx-teal:      #14b8a6;
  --rx-shadow-1:  0 1px 2px rgba(15,23,42,.04), 0 0 0 1px rgba(15,23,42,.04);
  --rx-shadow-2:  0 4px 16px -4px rgba(15,23,42,.10), 0 0 0 1px rgba(15,23,42,.05);
  --rx-shadow-3:  0 22px 60px -22px rgba(15,23,42,.45), 0 0 0 1px rgba(15,23,42,.06);
  --rx-radius:    14px;
  --rx-radius-sm: 10px;
  --rx-easing:    cubic-bezier(.2,.7,.2,1);
}

/* ---------------- Frame ---------------- */
.gradio-container {
  max-width: 1320px !important;
  margin: 0 auto !important;
  font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-feature-settings: 'cv11', 'ss01', 'ss03';
  letter-spacing: -0.005em;
}
* { box-sizing: border-box; }

/* ---------------- Hero ---------------- */
.rx-hero {
  position: relative;
  padding: 40px 36px 36px;
  margin: 14px 0 22px;
  border-radius: 22px;
  background:
    radial-gradient(900px 200px at -10% -20%, rgba(99,102,241,.55), transparent 60%),
    radial-gradient(900px 240px at 110% -10%, rgba(20,184,166,.45), transparent 60%),
    radial-gradient(700px 200px at 50% 120%, rgba(139,92,246,.35), transparent 60%),
    linear-gradient(135deg, #0b1020 0%, #131b39 60%, #1d1240 100%);
  color: #f8fafc;
  overflow: hidden;
  box-shadow: var(--rx-shadow-3);
}
.rx-hero::before {
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
  background-size: 32px 32px;
  mask-image: radial-gradient(800px 240px at 50% 0%, #000, transparent 80%);
  opacity: .4;
}
.rx-hero h1 {
  font-size: 36px; font-weight: 800; letter-spacing: -0.03em;
  margin: 0 0 8px; line-height: 1.05;
  background: linear-gradient(95deg,#fff 0%,#c7d2fe 45%,#67e8f9 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.rx-hero p {
  margin: 0; opacity: .82; font-size: 14.5px; line-height: 1.55;
  max-width: 780px; font-weight: 400;
}
.rx-hero .pills { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.rx-hero .pill {
  background: rgba(255,255,255,.06); backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,.12);
  padding: 6px 14px; border-radius: 999px; font-size: 12px;
  letter-spacing: .005em; font-weight: 500;
  display: inline-flex; align-items: center; gap: 6px;
  transition: all .25s var(--rx-easing);
}
.rx-hero .pill:hover { transform: translateY(-1px); border-color: rgba(255,255,255,.22); }
.rx-hero .pill.ok   { background: rgba(16,185,129,.16); border-color: rgba(16,185,129,.4); color: #a7f3d0; }
.rx-hero .pill.warn { background: rgba(245,158,11,.16); border-color: rgba(245,158,11,.4); color: #fde68a; }
.rx-hero .pill.err  { background: rgba(239,68,68,.16);  border-color: rgba(239,68,68,.4);  color: #fecaca; }
.rx-hero .pill .dot {
  width: 6px; height: 6px; border-radius: 50%; background: currentColor;
  box-shadow: 0 0 0 3px rgba(255,255,255,.06);
}

/* ---------------- Card ---------------- */
.rx-card {
  background: var(--rx-bg-0);
  border: 1px solid var(--rx-line);
  border-radius: var(--rx-radius);
  padding: 18px 20px;
  box-shadow: var(--rx-shadow-1);
  transition: all .2s var(--rx-easing);
}
.rx-card:hover { box-shadow: var(--rx-shadow-2); }
.rx-card .h {
  font-weight: 700; color: var(--rx-ink); font-size: 14px;
  letter-spacing: -.01em; margin: 0 0 4px;
  display: flex; align-items: center; gap: 8px;
}
.rx-card .h .num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 6px;
  background: var(--rx-brand-50); color: var(--rx-brand);
  font-size: 11px; font-weight: 800;
}
.rx-card .sub { color: var(--rx-ink-3); font-size: 13px; line-height: 1.5; margin: 0; }

/* ---------------- KPI tiles ---------------- */
.rx-kpi-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin: 4px 0 18px;
}
.rx-kpi {
  position: relative;
  padding: 16px 18px; border-radius: 14px;
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  color: #fff; box-shadow: var(--rx-shadow-2);
  overflow: hidden; isolation: isolate;
  transition: transform .25s var(--rx-easing), box-shadow .25s var(--rx-easing);
}
.rx-kpi:hover { transform: translateY(-2px); }
.rx-kpi::before {
  content: ''; position: absolute; inset: 0;
  background:
    radial-gradient(220px 80px at 90% 0%, rgba(255,255,255,.22), transparent 55%),
    radial-gradient(120px 60px at 0% 100%, rgba(0,0,0,.18), transparent 60%);
  z-index: -1;
}
.rx-kpi .v {
  font-size: 30px; font-weight: 800; letter-spacing: -.025em;
  line-height: 1.04; margin: 0 0 2px;
  font-variant-numeric: tabular-nums;
}
.rx-kpi .l {
  font-size: 10.5px; text-transform: uppercase;
  letter-spacing: .1em; opacity: .82; font-weight: 700;
}
.rx-kpi .delta { font-size: 11px; opacity: .9; margin-top: 6px; font-weight: 500; }
.rx-kpi.green   { background: linear-gradient(135deg,#10b981 0%,#059669 100%); }
.rx-kpi.amber   { background: linear-gradient(135deg,#f59e0b 0%,#d97706 100%); }
.rx-kpi.red     { background: linear-gradient(135deg,#ef4444 0%,#b91c1c 100%); }
.rx-kpi.violet  { background: linear-gradient(135deg,#8b5cf6 0%,#6d28d9 100%); }
.rx-kpi.slate   { background: linear-gradient(135deg,#475569 0%,#1e293b 100%); }
.rx-kpi.teal    { background: linear-gradient(135deg,#14b8a6 0%,#0f766e 100%); }

/* ---------------- Score bar ---------------- */
.rx-bar {
  background: var(--rx-bg-2); border-radius: 999px; height: 6px;
  overflow: hidden; position: relative;
}
.rx-bar > i {
  display: block; height: 100%;
  background: linear-gradient(90deg, var(--rx-brand-2), var(--rx-brand));
  border-radius: 999px; transition: width .6s var(--rx-easing);
  box-shadow: 0 0 10px -2px var(--rx-brand-2);
}
.rx-bar.high > i  { background: linear-gradient(90deg, #34d399, #10b981); box-shadow: 0 0 10px -2px #10b981; }
.rx-bar.mid > i   { background: linear-gradient(90deg, #fbbf24, #f59e0b); box-shadow: 0 0 10px -2px #f59e0b; }
.rx-bar.low > i   { background: linear-gradient(90deg, #fb7185, #ef4444); box-shadow: 0 0 10px -2px #ef4444; }

/* ---------------- Section trace ---------------- */
.rx-trace { margin-bottom: 14px; }
.rx-trace .head {
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; margin-bottom: 10px;
}
.rx-trace .label {
  font-weight: 700; color: var(--rx-ink); font-size: 13.5px;
  letter-spacing: -.005em;
}
.rx-trace .meta { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.rx-before, .rx-after {
  border-radius: 11px; padding: 12px 14px; font-size: 13.5px; line-height: 1.6;
  margin-top: 6px; transition: all .2s var(--rx-easing);
}
.rx-before {
  background: linear-gradient(135deg, #fff7ed, #fffbeb);
  border-left: 3px solid #f59e0b; color: #5a3a05;
}
.rx-after {
  background: linear-gradient(135deg, #ecfdf5, #f0fdfa);
  border-left: 3px solid #10b981; color: #064e3b;
}
.rx-before:hover, .rx-after:hover { transform: translateX(2px); }
.rx-before .lbl, .rx-after .lbl {
  text-transform: uppercase; font-size: 9.5px; font-weight: 800;
  letter-spacing: .1em; opacity: .55; margin-right: 8px;
}
.rx-flags {
  margin-top: 10px; padding: 9px 13px; border-radius: 9px;
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  color: #78350f; font-size: 12.5px; line-height: 1.5;
  border-left: 3px solid #f59e0b;
}
.rx-trace-progress {
  display: flex; gap: 4px; margin-top: 10px; align-items: center;
}
.rx-trace-progress .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--rx-line-2); transition: all .3s var(--rx-easing);
}
.rx-trace-progress .dot.done { background: var(--rx-mint); }
.rx-trace-progress .dot.accept { background: var(--rx-brand); box-shadow: 0 0 0 3px var(--rx-brand-50); }
.rx-trace-progress .dim {
  font-size: 11px; color: var(--rx-ink-3); margin-left: 6px;
}

/* ---------------- Score pills + tags ---------------- */
.rx-pill {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-weight: 700; font-size: 11px; color: #fff;
  letter-spacing: .03em; font-variant-numeric: tabular-nums;
}
.rx-pill.high   { background: linear-gradient(135deg,#10b981,#059669); }
.rx-pill.mid    { background: linear-gradient(135deg,#f59e0b,#d97706); }
.rx-pill.low    { background: linear-gradient(135deg,#ef4444,#b91c1c); }
.rx-pill.ghost  { background: var(--rx-bg-2); color: var(--rx-ink-2); }
.rx-pill.iter   { background: #ede9fe; color: #5b21b6; }
.rx-pill.changed { background: linear-gradient(135deg,#6366f1,#4f46e5); }
.rx-pill.unchanged { background: var(--rx-line); color: var(--rx-ink-2); }

.rx-tag {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  font-size: 11.5px; font-weight: 500; margin: 2px 4px 2px 0;
  background: var(--rx-brand-50); color: var(--rx-brand);
  border: 1px solid #e0e7ff;
  transition: all .15s var(--rx-easing);
}
.rx-tag:hover { transform: translateY(-1px); }
.rx-tag.miss { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.rx-tag.match { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }

/* ---------------- Hiring-manager review (single-focus) ---------------- */
.rx-review-hero {
  background:
    radial-gradient(600px 140px at 0% 0%, rgba(99,102,241,.10), transparent 60%),
    radial-gradient(500px 140px at 100% 0%, rgba(20,184,166,.10), transparent 60%),
    var(--rx-bg-0);
  border: 1px solid var(--rx-line); border-radius: var(--rx-radius);
  padding: 28px 30px; margin-bottom: 14px;
  box-shadow: var(--rx-shadow-2);
}
.rx-review-hero .top {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 14px; flex-wrap: wrap;
}
.rx-review-hero h2 {
  margin: 0 0 4px; font-size: 22px; font-weight: 800;
  letter-spacing: -.025em; color: var(--rx-ink);
}
.rx-review-hero .subtitle {
  color: var(--rx-ink-3); font-size: 13.5px; margin: 0;
}
.rx-review-hero .scoreBox {
  text-align: right; min-width: 130px;
}
.rx-review-hero .scoreBox .num {
  font-size: 42px; font-weight: 800; letter-spacing: -.04em;
  line-height: 1; font-variant-numeric: tabular-nums;
  background: linear-gradient(135deg,#4f46e5,#8b5cf6);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.rx-review-hero .scoreBox .num.high { background: linear-gradient(135deg,#10b981,#059669); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.rx-review-hero .scoreBox .num.mid  { background: linear-gradient(135deg,#f59e0b,#d97706); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.rx-review-hero .scoreBox .num.low  { background: linear-gradient(135deg,#ef4444,#b91c1c); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.rx-review-hero .scoreBox .label {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .1em;
  color: var(--rx-ink-4); font-weight: 700; margin-top: 4px;
}
.rx-review-hero .verdict {
  margin-top: 18px; padding: 14px 16px; border-radius: 11px;
  background: var(--rx-bg-1); border-left: 3px solid var(--rx-brand);
  font-style: italic; color: var(--rx-ink-2); font-size: 14px; line-height: 1.55;
}
.rx-review-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px;
}
.rx-review-grid .col {
  background: var(--rx-bg-0); border: 1px solid var(--rx-line);
  border-radius: var(--rx-radius); padding: 16px 18px;
  box-shadow: var(--rx-shadow-1);
}
.rx-review-grid .col h5 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
  margin: 0 0 10px; color: var(--rx-ink-4); font-weight: 800;
  display: flex; align-items: center; gap: 6px;
}
.rx-review-grid .col h5::before {
  content: ''; width: 4px; height: 14px; border-radius: 2px;
  background: var(--rx-brand);
}
.rx-review-grid .col.weak h5::before { background: var(--rx-amber); }
.rx-review-grid .col ul {
  margin: 0; padding-left: 16px; font-size: 13.5px; line-height: 1.65;
  color: var(--rx-ink); list-style-type: none;
}
.rx-review-grid .col ul li {
  position: relative; padding-left: 14px; margin-bottom: 6px;
}
.rx-review-grid .col ul li::before {
  content: '✓'; position: absolute; left: 0; color: var(--rx-mint); font-weight: 800;
}
.rx-review-grid .col.weak ul li::before { content: '!'; color: var(--rx-amber); }
.rx-keywords-card {
  margin-top: 14px;
  background: var(--rx-bg-0); border: 1px solid var(--rx-line);
  border-radius: var(--rx-radius); padding: 16px 18px;
  box-shadow: var(--rx-shadow-1);
}
.rx-keywords-card h5 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
  margin: 0 0 10px; color: var(--rx-ink-4); font-weight: 800;
}

/* ---------------- Tables (JD) ---------------- */
.rx-table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  font-size: 13.5px; background: var(--rx-bg-0);
  border-radius: 10px; overflow: hidden;
  box-shadow: var(--rx-shadow-1);
}
.rx-table th, .rx-table td {
  padding: 12px 14px; text-align: left;
  border-bottom: 1px solid var(--rx-line);
}
.rx-table th {
  background: var(--rx-bg-1); color: var(--rx-ink-3); font-weight: 700;
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em;
}
.rx-table tr:last-child td { border-bottom: none; }
.rx-table tr:hover td { background: var(--rx-bg-1); }
.rx-table td.gap { color: var(--rx-ink-3); font-size: 12.5px; }
.rx-table td.num { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--rx-ink); }

/* ---------------- Setup ---------------- */
.rx-setup {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 14px; margin-top: 8px;
}
.rx-setup .opt {
  position: relative; background: var(--rx-bg-0);
  border: 1px solid var(--rx-line); border-radius: var(--rx-radius);
  padding: 22px 22px 18px; box-shadow: var(--rx-shadow-1);
  transition: all .2s var(--rx-easing);
}
.rx-setup .opt:hover { box-shadow: var(--rx-shadow-2); transform: translateY(-2px); }
.rx-setup .opt.ready {
  border-color: #86efac;
  background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
}
.rx-setup .opt.ready::before {
  content: '✓ READY';
  position: absolute; top: 14px; right: 14px;
  font-size: 10px; font-weight: 800; letter-spacing: .12em;
  color: #fff; background: linear-gradient(135deg,#10b981,#059669);
  padding: 4px 10px; border-radius: 999px;
  box-shadow: 0 4px 10px -2px rgba(16,185,129,.5);
}
.rx-setup .opt h4 {
  margin: 0 0 6px; font-size: 16px; color: var(--rx-ink);
  font-weight: 700; letter-spacing: -.015em;
}
.rx-setup .opt .why { color: var(--rx-ink-3); font-size: 13px; margin: 0 0 12px; line-height: 1.5; }
.rx-setup .opt .steps {
  background: linear-gradient(135deg, #0b1020, #131b39);
  color: #e2e8f0; padding: 14px 16px;
  border-radius: 10px; font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  font-size: 12px; line-height: 1.7;
  white-space: pre-wrap; word-break: break-word;
  border: 1px solid rgba(255,255,255,.06);
}
.rx-setup .opt .cost {
  display: inline-block; margin-top: 10px;
  font-size: 10.5px; color: var(--rx-ink-4); font-weight: 700;
  text-transform: uppercase; letter-spacing: .1em;
}

/* ---------------- Progress log ---------------- */
.rx-log {
  background: linear-gradient(135deg, #0b1020, #131b39);
  color: #cbd5e1;
  border-radius: 12px; padding: 16px 18px;
  font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  font-size: 12px; line-height: 1.75; max-height: 420px; overflow-y: auto;
  border: 1px solid rgba(255,255,255,.05);
  box-shadow: inset 0 0 50px rgba(0,0,0,.2);
}
.rx-log .stage  { color: #93c5fd; font-weight: 700; }
.rx-log .ok     { color: #6ee7b7; }
.rx-log .warn   { color: #fcd34d; }
.rx-log .err    { color: #fca5a5; }
.rx-log .muted  { color: #64748b; }
.rx-log::-webkit-scrollbar { width: 8px; }
.rx-log::-webkit-scrollbar-thumb { background: rgba(255,255,255,.08); border-radius: 4px; }
.rx-log::-webkit-scrollbar-track { background: transparent; }

/* ---------------- Footer + empty + misc ---------------- */
.rx-foot {
  text-align: center; color: var(--rx-ink-4);
  font-size: 12px; margin: 22px 0 8px;
}
.rx-empty {
  text-align: center; color: var(--rx-ink-3); padding: 36px 22px;
  background: var(--rx-bg-1);
  border: 1px dashed var(--rx-line-2);
  border-radius: var(--rx-radius); font-size: 13.5px; line-height: 1.5;
}
.rx-empty .big {
  font-size: 44px; margin-bottom: 8px; opacity: .35;
  display: inline-block;
}

/* Gradio overrides */
button.rx-cta {
  font-weight: 700 !important; letter-spacing: -.005em !important;
  font-size: 14px !important;
  padding: 13px 22px !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
  border: none !important; color: #fff !important;
  box-shadow: 0 6px 18px -4px rgba(79,70,229,.5) !important;
  transition: all .2s var(--rx-easing) !important;
}
button.rx-cta:hover { transform: translateY(-1px) !important; box-shadow: 0 10px 24px -4px rgba(79,70,229,.6) !important; }
"""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _score_class(score: float) -> str:
    if score >= 80: return "high"
    if score >= 60: return "mid"
    return "low"


def _kpi(label: str, value: str, klass: str = "", delta: str = "") -> str:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (
        f'<div class="rx-kpi {klass}">'
        f'<div class="v">{value}</div>'
        f'<div class="l">{label}</div>'
        f'{delta_html}'
        '</div>'
    )


def _empty(icon: str, msg: str) -> str:
    return f'<div class="rx-empty"><div class="big">{icon}</div>{msg}</div>'


def _bar(score: float, max_score: float = 100.0) -> str:
    pct = max(0.0, min(100.0, (score / max_score) * 100.0)) if max_score else 0.0
    cls = _score_class(score if max_score == 100 else (score / max_score) * 100.0)
    return f'<div class="rx-bar {cls}"><i style="width:{pct:.0f}%"></i></div>'


# ----------------------------------------------------------------------
# Hero / status
# ----------------------------------------------------------------------
def _hero_html() -> str:
    statuses = detect_available_backends()
    pills = []
    for bid, info in statuses.items():
        klass = "ok" if info["ready"] else ""
        label = info["name"].split(" (")[0]
        pills.append(
            f'<span class="pill {klass}"><span class="dot"></span>{label}</span>'
        )
    pills_html = "".join(pills)
    return (
        '<div class="rx-hero">'
        '<h1>Resume Enhancer · Multi-Agent Edition</h1>'
        '<p>A 7-agent pipeline that extracts every field from your <b>.tex</b> resume, '
        'restructures it for the target role, rewrites each bullet under deterministic '
        'safety guards, then validates against curated job descriptions and simulates '
        'a hiring manager read.</p>'
        f'<div class="pills">{pills_html}'
        '<span class="pill"><span class="dot"></span>7 agents</span>'
        '<span class="pill"><span class="dot"></span>Critic loop</span>'
        '<span class="pill"><span class="dot"></span>Overleaf-ready</span>'
        '</div>'
        '</div>'
    )


# ----------------------------------------------------------------------
# Result panels
# ----------------------------------------------------------------------
def _summary_html(result: PipelineResult) -> str:
    if result.status == "error":
        errs = "<br>".join(result.errors) or "Unknown error."
        return (
            f'<div class="rx-card" style="border-color:#fecaca;background:#fef2f2">'
            f'<div class="h" style="color:#991b1b">Pipeline failed</div>'
            f'<div class="sub" style="color:#7f1d1d">{errs}</div></div>'
        )
    sections_changed = sum(1 for t in result.section_traces if t.changed)
    sections_total = len(result.section_traces)
    elapsed = f"{result.elapsed_ms / 1000:.1f}s"
    ats = result.ats.score if result.ats else 0.0
    review_score = result.role_reviews[0].overall_score if result.role_reviews else 0.0
    jd_delta = result.jd_report.avg_delta if result.jd_report else 0.0
    jd_after = result.jd_report.avg_score_after if result.jd_report else 0.0
    delta_str = f"{jd_delta:+.1f} vs original" if jd_delta else ""

    backend_pretty = {
        "claude_code": "Claude Code login",
        "anthropic": "Anthropic API",
        "huggingface": "Hugging Face",
    }.get(result.backend, result.backend)

    role_pretty = ROLES.get(result.role, result.role)

    kpis = "".join([
        _kpi("Status", result.status.upper(), "green" if result.status == "complete" else "amber"),
        _kpi("Sections Improved", f"{sections_changed}/{sections_total}", "violet"),
        _kpi("ATS Score", f"{ats:.0f}/100", _score_class(ats)),
        _kpi("Hiring Manager", f"{review_score:.0f}/100", _score_class(review_score), f"target: {role_pretty}"),
        _kpi("JD Match", f"{jd_after:.0f}/100", "teal", delta_str),
        _kpi("Elapsed", elapsed, "slate", f"backend: {backend_pretty}"),
    ])
    warnings = ""
    if result.warnings:
        ws = "".join(f"<li>{w}</li>" for w in result.warnings)
        warnings = (
            '<div class="rx-card" style="margin-top:12px;border-color:#fde68a;background:#fffbeb">'
            '<div class="h" style="color:#92400e">Notes</div>'
            f'<ul style="margin:6px 0 0 18px;color:#78350f;font-size:13px">{ws}</ul></div>'
        )
    return f'<div class="rx-kpi-row">{kpis}</div>{warnings}'


def _sections_html(result: PipelineResult) -> str:
    if not result.section_traces:
        return _empty("◔", "Per-section before / after traces will appear here once the pipeline runs.")
    rows = []
    for t in result.section_traces:
        score_class = _score_class(t.final_score)
        flags = ""
        if t.iterations:
            last = t.iterations[-1]
            if last.violations:
                flags = (
                    '<div class="rx-flags"><b>Critic flags:</b> '
                    + "; ".join(last.violations[:4]) + '</div>'
                )
        # iteration progress dots
        iter_dots = []
        for s in t.iterations:
            cls = "accept" if s.accepted else ("done" if s.verdict != "error" else "")
            iter_dots.append(f'<span class="dot {cls}" title="iter {s.iteration}: {s.score:.0f}"></span>')
        progress_block = (
            '<div class="rx-trace-progress">'
            + "".join(iter_dots)
            + f'<span class="dim">{t.iterations_used} iter · final {t.final_score:.0f}/100</span>'
            + '</div>'
        ) if iter_dots else ""
        note = (
            f'<div style="font-size:12px;color:var(--rx-ink-3);margin-top:6px;font-style:italic">{t.note}</div>'
            if t.note else ""
        )
        change_pill = (
            '<span class="rx-pill changed">CHANGED</span>'
            if t.changed else
            '<span class="rx-pill unchanged">UNCHANGED</span>'
        )
        rows.append(
            '<div class="rx-card rx-trace">'
            '<div class="head">'
            f'<div class="label">{t.label}</div>'
            f'<div class="meta">'
            f'<span class="rx-pill {score_class}">{t.final_score:.0f}</span>'
            f'{change_pill}'
            '</div></div>'
            f'<div class="rx-before"><span class="lbl">Before</span>{t.before}</div>'
            f'<div class="rx-after"><span class="lbl">After</span>{t.after}</div>'
            f'{flags}{progress_block}{note}'
            '</div>'
        )
    return "\n".join(rows)


def _review_html(result: PipelineResult) -> str:
    if not result.role_reviews:
        return _empty("☼", "Hiring-manager review is off, or the pipeline did not complete.")
    # Single-focus card for the target role.
    r = result.role_reviews[0]
    score_class = _score_class(r.overall_score)
    strengths = "".join(f"<li>{s}</li>" for s in r.strengths) or "<li><i>none returned</i></li>"
    weaknesses = "".join(f"<li>{w}</li>" for w in r.weaknesses) or "<li><i>none</i></li>"
    missing = "".join(f'<span class="rx-tag miss">{k}</span>' for k in r.missing_keywords) or '<span class="rx-tag">none</span>'

    hero = (
        '<div class="rx-review-hero">'
        '<div class="top">'
        '<div>'
        f'<h2>{r.role_name}</h2>'
        '<p class="subtitle">Simulated hiring-manager read against this role profile</p>'
        '</div>'
        '<div class="scoreBox">'
        f'<div class="num {score_class}">{r.overall_score:.0f}</div>'
        '<div class="label">Phone-screen<br/>likelihood</div>'
        '</div>'
        '</div>'
        f'<div style="margin-top:18px">{_bar(r.overall_score)}</div>'
        f'<div class="verdict">"{r.one_line_verdict}"</div>'
        '</div>'
    )
    grid = (
        '<div class="rx-review-grid">'
        f'<div class="col"><h5>Strengths</h5><ul>{strengths}</ul></div>'
        f'<div class="col weak"><h5>Weaknesses</h5><ul>{weaknesses}</ul></div>'
        '</div>'
    )
    keywords = (
        '<div class="rx-keywords-card">'
        '<h5>Missing high-impact keywords</h5>'
        f'{missing}'
        '</div>'
    )
    # Optional cross-role mini-cards if more than one review was run
    extra = ""
    if len(result.role_reviews) > 1:
        chips = []
        for rr in result.role_reviews[1:]:
            kc = _score_class(rr.overall_score)
            chips.append(
                '<div class="rx-card" style="display:flex;justify-content:space-between;'
                'align-items:center;padding:14px 18px">'
                f'<div><b>{rr.role_name}</b><div style="font-size:12.5px;color:var(--rx-ink-3);'
                f'margin-top:2px">"{rr.one_line_verdict}"</div></div>'
                f'<span class="rx-pill {kc}">{rr.overall_score:.0f}</span></div>'
            )
        extra = (
            '<div style="margin-top:18px"><h5 style="font-size:11px;text-transform:uppercase;'
            'letter-spacing:.08em;color:var(--rx-ink-4);margin:0 0 10px;font-weight:800">'
            'Cross-role validation</h5>'
            + "".join(chips) + "</div>"
        )
    return hero + grid + keywords + extra


def _jd_html(result: PipelineResult) -> str:
    bits = []
    if result.jd_report and result.jd_report.samples:
        bits.append(_jd_table(result.jd_report, "Target role"))
    for r in result.cross_role_jd_reports:
        bits.append(_jd_table(r, "Cross-validation"))
    if not bits:
        return _empty("◔", "JD matching is off, or no JDs are loaded for this role.")
    return "\n\n".join(bits)


def _jd_table(report, kind: str) -> str:
    role_pretty = ROLES.get(report.role_id, report.role_id)
    rows_html = [
        "<tr><th>Job Description</th><th>Archetype</th>"
        "<th class='num'>Before</th><th class='num'>After</th>"
        "<th class='num'>Δ</th><th>Top gaps</th></tr>"
    ]
    for s in report.samples:
        klass = "high" if s.delta >= 5 else ("mid" if s.delta >= 0 else "low")
        gaps = ", ".join(s.missing_keywords[:5]) or "—"
        rows_html.append(
            f'<tr><td><b>{s.title}</b></td>'
            f'<td><span class="rx-tag">{s.company_archetype}</span></td>'
            f'<td class="num">{s.score_before:.0f}</td>'
            f'<td class="num">{s.score_after:.0f}</td>'
            f'<td><span class="rx-pill {klass}">{s.delta:+.1f}</span></td>'
            f'<td class="gap">{gaps}</td></tr>'
        )
    table = '<table class="rx-table">' + "".join(rows_html) + "</table>"
    delta_class = "high" if report.avg_delta >= 5 else ("mid" if report.avg_delta >= 0 else "low")
    avg_block = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:14px;flex-wrap:wrap;gap:8px">'
        f'<div><b style="font-size:15px;letter-spacing:-.015em">{role_pretty}</b> '
        f'<span class="rx-tag" style="margin-left:6px">{kind}</span></div>'
        f'<div style="font-size:13px;color:var(--rx-ink-3)">'
        f'avg before/after: <b style="color:var(--rx-ink)">{report.avg_score_before:.1f}</b> → '
        f'<b style="color:var(--rx-ink)">{report.avg_score_after:.1f}</b> '
        f'<span class="rx-pill {delta_class}" style="margin-left:6px">{report.avg_delta:+.1f}</span>'
        f'</div></div>'
    )
    gaps_block = ""
    if report.top_gaps:
        gaps = "".join(f'<span class="rx-tag miss">{k}</span>' for k in report.top_gaps)
        gaps_block = (
            f'<div style="margin-top:14px"><h5 style="font-size:10.5px;text-transform:uppercase;'
            f'letter-spacing:.1em;color:var(--rx-ink-4);margin:0 0 8px;font-weight:800">'
            f'Top cross-JD gaps</h5>{gaps}</div>'
        )
    return f'<div class="rx-card">{avg_block}{table}{gaps_block}</div>'


# ----------------------------------------------------------------------
# Setup tab
# ----------------------------------------------------------------------
def _setup_html() -> str:
    statuses = detect_available_backends()
    intro = (
        '<div class="rx-card" style="margin-bottom:14px">'
        '<div class="h"><span class="num">i</span>Backend setup</div>'
        '<div class="sub">Pick whichever option is easiest. The app auto-detects '
        'whatever you have configured. You can also override the backend per '
        'enhancement in the Enhance tab.</div></div>'
    )
    instructions = {
        "claude_code": (
            "# Recommended if you already use Claude Code\n"
            "# 1. Make sure the Claude Code CLI is installed\n"
            "# 2. Log in once:\n"
            "claude /login\n\n"
            "# 3. Restart this app. Done — no API key needed.\n"
            "#    The app uses your Claude subscription automatically."
        ),
        "anthropic": (
            "# Highest quality. Pay-per-token (cents per resume).\n"
            "# 1. Get an API key:\n"
            "#    https://console.anthropic.com/settings/keys\n\n"
            "# 2. Set the key (one of):\n"
            "export ANTHROPIC_API_KEY='sk-ant-...'           # macOS / Linux\n"
            "$env:ANTHROPIC_API_KEY='sk-ant-...'              # PowerShell\n\n"
            "# 3. Restart the app."
        ),
        "huggingface": (
            "# Free tier. Lower quality, but $0.\n"
            "# 1. Get a free token:\n"
            "#    https://huggingface.co/settings/tokens\n\n"
            "# 2. Set the token:\n"
            "export HF_API_KEY='hf_...'                       # macOS / Linux\n"
            "$env:HF_API_KEY='hf_...'                          # PowerShell\n\n"
            "# 3. Restart the app."
        ),
    }
    cards = []
    for bid, info in statuses.items():
        klass = "ready" if info["ready"] else ""
        cards.append(
            f'<div class="opt {klass}">'
            f'<h4>{info["name"]}</h4>'
            f'<p class="why">{info["needs"]}</p>'
            f'<div class="steps">{instructions[bid]}</div>'
            f'<span class="cost">{info["cost"]}</span>'
            f'</div>'
        )
    grid = '<div class="rx-setup">' + "".join(cards) + "</div>"
    best = best_available_backend()
    if best:
        foot = (
            f'<div class="rx-card" style="margin-top:14px;border-color:#86efac;background:#f0fdf4">'
            f'<div class="h" style="color:#065f46"><span class="num" style="background:#bbf7d0;color:#065f46">✓</span>'
            f'Auto-detected backend: '
            f'<code style="background:#fff;padding:3px 10px;border-radius:6px;font-size:12px">{best}</code></div>'
            f'<div class="sub" style="color:#047857">Ready to enhance. '
            f'Head to the <b>Enhance</b> tab.</div></div>'
        )
    else:
        foot = (
            '<div class="rx-card" style="margin-top:14px;border-color:#fecaca;background:#fef2f2">'
            '<div class="h" style="color:#991b1b"><span class="num" style="background:#fecaca;color:#991b1b">!</span>'
            'No backend configured yet</div>'
            '<div class="sub" style="color:#7f1d1d">Pick one of the options above. '
            'Claude Code is the simplest if you already use it.</div></div>'
        )
    return intro + grid + foot


# ----------------------------------------------------------------------
# Enhance handler (with progressive UI)
# ----------------------------------------------------------------------
def _backend_choices() -> list[tuple[str, str]]:
    return [
        ("Auto-detect (recommended)", "auto"),
        ("Claude Code (uses your login)", "claude_code"),
        ("Anthropic API (paid)", "anthropic"),
        ("Hugging Face (free)", "huggingface"),
    ]


def _enhance_handler(
    file, role_id: str, backend: str,
    enable_critic: bool, enable_review: bool, enable_cross: bool, max_iter: int,
):
    blank_outputs = (
        _empty("◌", "Per-section before / after will appear here."),
        _empty("☼", "Hiring-manager review will appear here."),
        _empty("◌", "JD scores will appear here."),
        None, "",
        gr.update(visible=False),
    )

    if file is None:
        yield (
            _empty("↑", "Upload a <b>.tex</b> resume on the left and pick a target role."),
            *blank_outputs,
        )
        return

    tex_path = Path(file.name if hasattr(file, "name") else file)
    if tex_path.suffix.lower() != ".tex":
        yield (
            _empty("⚠", "Please upload a <b>.tex</b> file. PDFs are not supported in this build."),
            *blank_outputs,
        )
        return

    if backend != "auto" and not is_backend_configured(backend):
        yield (
            _empty(
                "⚠",
                f"Backend <b>{backend}</b> is not configured. "
                "Visit the <b>Setup</b> tab for instructions, or change to "
                "<b>Auto-detect</b>."
            ),
            *blank_outputs,
        )
        return

    if backend == "auto" and best_available_backend() is None:
        yield (
            _empty(
                "○",
                "No backend is configured. Open the <b>Setup</b> tab — "
                "the simplest path is Claude Code if you're already logged in."
            ),
            *blank_outputs,
        )
        return

    cfg = PipelineConfig(
        role_id=role_id, backend=backend,
        enable_critic=enable_critic,
        enable_role_review=enable_review,
        enable_jd_matching=True,
        enable_cross_role=enable_cross,
        max_iterations=int(max_iter),
    )

    q: "queue.Queue" = queue.Queue()
    holder: dict = {"result": None, "error": None}

    def progress(event: str, data: dict) -> None:
        try:
            q.put((event, data), timeout=1.0)
        except queue.Full:
            pass

    def worker() -> None:
        try:
            holder["result"] = run_pipeline(tex_path, cfg, progress=progress)
        except Exception as e:                              # noqa: BLE001
            log.exception("[ui] pipeline crashed")
            holder["error"] = str(e)
        finally:
            q.put(("done", {}))

    t = threading.Thread(target=worker, daemon=True, name="enhance-worker")
    t.start()

    progress_lines: list[str] = ['<span class="stage">[start]</span> spinning up agents…']
    last_emit = time.monotonic()
    while True:
        try:
            event, data = q.get(timeout=0.6)
        except queue.Empty:
            if time.monotonic() - last_emit > 1.2:
                yield (
                    _progress_html(progress_lines),
                    *blank_outputs[:-3],
                    None, "",
                    gr.update(visible=False),
                )
                last_emit = time.monotonic()
            continue
        if event == "done":
            break
        progress_lines.append(_format_event(event, data))
        last_emit = time.monotonic()
        yield (
            _progress_html(progress_lines),
            *blank_outputs[:-3],
            None, "",
            gr.update(visible=False),
        )

    if holder["error"]:
        yield (
            f'<div class="rx-card" style="border-color:#fecaca;background:#fef2f2">'
            f'<div class="h" style="color:#991b1b">Pipeline crashed</div>'
            f'<div class="sub" style="color:#7f1d1d">{holder["error"]}</div></div>',
            *blank_outputs,
        )
        return
    result: PipelineResult = holder["result"]
    yield (
        _summary_html(result),
        _sections_html(result),
        _review_html(result),
        _jd_html(result),
        result.tex_path,
        result.tex_content,
        gr.update(visible=True),
    )


def _progress_html(lines: list[str]) -> str:
    items = "".join(f"<div>{ln}</div>" for ln in lines[-30:])
    return (
        '<div class="rx-card">'
        '<div class="h"><span class="num">▸</span>Live agent trace</div>'
        f'<div class="rx-log" style="margin-top:10px">{items}</div></div>'
    )


def _format_event(event: str, data: dict) -> str:
    if event == "stage":
        name = data.get("name", "")
        status = data.get("status", "")
        css = "ok" if status == "done" else "stage"
        extras = []
        for k, v in data.items():
            if k in ("name", "status"):
                continue
            extras.append(f"{k}={v}")
        suffix = f' <span class="muted">{", ".join(extras)}</span>' if extras else ""
        return f'<span class="stage">[{name}]</span> <span class="{css}">{status}</span>{suffix}'
    if event == "section":
        label = data.get("label", "")
        status = data.get("status", "")
        score = data.get("score")
        if score is not None:
            cls = "ok" if score >= 80 else ("warn" if score >= 60 else "err")
            return f'  <span class="muted">└</span> <span class="{cls}">{status}</span> {label} <span class="muted">({score:.0f})</span>'
        return f'  <span class="muted">├ {status}: {label}</span>'
    if event == "review":
        return f'  <span class="stage">└ role-review</span> {data.get("status","")}: {data.get("role","")}'
    return f"{event}: {data}"


# ----------------------------------------------------------------------
# Build the Gradio app
# ----------------------------------------------------------------------
def build_app() -> gr.Blocks:
    role_choices = [(name, rid) for rid, name in ROLES.items()]
    with gr.Blocks(title="Resume Enhancer · Multi-Agent Edition") as app:
        gr.HTML(_hero_html())

        with gr.Tabs():
            # ---------- Enhance ----------
            with gr.Tab("Enhance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div class="rx-card" style="margin-bottom:12px">'
                            '<div class="h"><span class="num">1</span>Upload your .tex resume</div>'
                            '<div class="sub">Output is also a .tex — open in '
                            '<a href="https://overleaf.com" target="_blank">Overleaf</a> to compile.</div></div>'
                        )
                        file_in = gr.File(
                            label="Resume (.tex)",
                            file_types=[".tex"],
                            file_count="single",
                            height=140,
                        )
                        gr.HTML(
                            '<div class="rx-card" style="margin:12px 0">'
                            '<div class="h"><span class="num">2</span>Pick a target role</div>'
                            '<div class="sub">The pipeline tunes section order, '
                            'keyword emphasis, and bullet phrasing for this role only.</div></div>'
                        )
                        role_in = gr.Dropdown(
                            choices=role_choices, value="ai_ml_engineer",
                            label="Target role",
                            interactive=True,
                        )
                        gr.HTML(
                            '<div class="rx-card" style="margin:12px 0">'
                            '<div class="h"><span class="num">3</span>Backend</div>'
                            '<div class="sub">Auto-detect picks Claude Code &gt; Anthropic &gt; HF '
                            'depending on what\'s configured. See the Setup tab.</div></div>'
                        )
                        backend_in = gr.Dropdown(
                            choices=_backend_choices(),
                            value="auto",
                            label="LLM backend",
                            interactive=True,
                        )
                        with gr.Accordion("Advanced options", open=False):
                            critic_in = gr.Checkbox(
                                value=True,
                                label="Enable critic loop (recommended)",
                            )
                            review_in = gr.Checkbox(
                                value=True,
                                label="Run hiring-manager simulation for the target role",
                            )
                            cross_in = gr.Checkbox(
                                value=False,
                                label="Also score against the other 4 roles (slower)",
                            )
                            iter_in = gr.Slider(
                                minimum=1, maximum=4, value=3, step=1,
                                label="Max iterations per section",
                            )
                        run_btn = gr.Button(
                            "Enhance Resume",
                            variant="primary", size="lg",
                            elem_classes=["rx-cta"],
                        )

                    with gr.Column(scale=2):
                        summary_out = gr.HTML(
                            _empty("↑", "Upload a <b>.tex</b> resume on the left, "
                                   "pick a target role, then click <b>Enhance Resume</b>.")
                        )

            # ---------- Sections ----------
            with gr.Tab("Sections"):
                gr.HTML(
                    '<div class="rx-card" style="margin-bottom:12px">'
                    '<div class="h"><span class="num">↦</span>Per-section before / after</div>'
                    '<div class="sub">Every bullet went through the Enhancer ↔ Critic loop. '
                    'You see the original, the rewrite, the critic\'s 5-dimension score, and any '
                    'safety guard interventions.</div></div>'
                )
                sections_out = gr.HTML(_empty("◌", "Section-level traces appear here after enhancement."))

            # ---------- Hiring-Manager Review ----------
            with gr.Tab("Hiring-Manager Review"):
                gr.HTML(
                    '<div class="rx-card" style="margin-bottom:12px">'
                    '<div class="h"><span class="num">☼</span>Hiring-manager simulation</div>'
                    '<div class="sub">A single-focus read against your <b>target role</b>. '
                    'The agent plays a senior hiring manager at a top-tier company for that role '
                    'and returns strengths, weaknesses, missing high-impact keywords, and a '
                    'phone-screen likelihood score.</div></div>'
                )
                review_out = gr.HTML(_empty("☼", "Hiring-manager review appears here after enhancement."))

            # ---------- JD Matching ----------
            with gr.Tab("JD Matching"):
                gr.HTML(
                    '<div class="rx-card" style="margin-bottom:12px">'
                    '<div class="h"><span class="num">≣</span>Curated job descriptions</div>'
                    '<div class="sub">5 hand-curated JDs for your target role with synonym lists. '
                    'We score the original and enhanced resumes against each, and report the lift.</div></div>'
                )
                jd_out = gr.HTML(_empty("◌", "JD scores against curated samples appear here."))

            # ---------- Download ----------
            with gr.Tab("Download"):
                with gr.Group(visible=False) as download_group:
                    gr.HTML(
                        '<div class="rx-card">'
                        '<div class="h"><span class="num">↓</span>Your enhanced resume — .tex output</div>'
                        '<div class="sub">Open the file in '
                        '<a href="https://overleaf.com" target="_blank">Overleaf</a> '
                        'to compile to PDF. The template uses standard packages: '
                        '<code style="background:var(--rx-bg-2);padding:2px 6px;border-radius:4px;font-size:12px">fontawesome5</code>, '
                        '<code style="background:var(--rx-bg-2);padding:2px 6px;border-radius:4px;font-size:12px">sourcesanspro</code>, '
                        '<code style="background:var(--rx-bg-2);padding:2px 6px;border-radius:4px;font-size:12px">tabularx</code>, '
                        '<code style="background:var(--rx-bg-2);padding:2px 6px;border-radius:4px;font-size:12px">enumitem</code>.</div></div>'
                    )
                    tex_file_out = gr.File(label="Enhanced .tex", interactive=False)
                    tex_text_out = gr.Code(
                        label="Preview (read-only)",
                        language="latex",
                        lines=24,
                        interactive=False,
                    )

            # ---------- Setup ----------
            with gr.Tab("Setup"):
                setup_out = gr.HTML(_setup_html())
                refresh_btn = gr.Button("Refresh status", size="sm")
                refresh_btn.click(lambda: _setup_html(), outputs=setup_out)

            # ---------- About ----------
            with gr.Tab("About"):
                gr.Markdown(_about_md())

        run_btn.click(
            _enhance_handler,
            inputs=[file_in, role_in, backend_in, critic_in, review_in, cross_in, iter_in],
            outputs=[
                summary_out, sections_out, review_out, jd_out,
                tex_file_out, tex_text_out, download_group,
            ],
        )

        gr.HTML(
            '<div class="rx-foot">v4 · Multi-agent skill-driven · '
            'Claude Code / Anthropic / Hugging Face · Output: .tex (Overleaf-ready)</div>'
        )
    return app


def _about_md() -> str:
    return """
### How it works

```
.tex input
   |
   v
+--------------+    +-------------+    +-------------+    +-------------+
| Parse (regex)| -> | Repair (LLM)| -> | Complete    | -> | Plan (LLM)  |
| -> ResumeIR  |    | (Extractor) |    | placeholders|    | section order
+--------------+    +-------------+    +-------------+    +-------------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | Per-section loop     |
                                                   | Enhancer <-> Critic  |
                                                   | (max 3 iter, >=82)   |
                                                   +----------------------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | Safety guards        |
                                                   | length + protected   |
                                                   +----------------------+
                                                              |
                          +-----------------------------------+--------------------+
                          v                                                        v
              +-------------------+                                  +-------------------+
              | Render -> .tex    |                                  | ATS / JD / Review |
              | (Jinja2 template) |                                  | TARGET role       |
              +-------------------+                                  +-------------------+
```

### Agents

| Agent | Reads | Writes | Failure mode |
|---|---|---|---|
| **Extractor** | parsed IR + raw .tex | repaired IR | falls through to parser output |
| **Completer** | IR | IR with `[PLACEHOLDER]` tokens for missing required fields | always succeeds (deterministic) |
| **Planner** | IR + role profile | section / skill / experience ordering | falls back to per-role default |
| **Enhancer** | one section + critique | rewritten section | returns "" → orchestrator keeps original |
| **Critic** | (before, after) | 5-dim score + verdict (JSON) | permissive accept on parse error |
| **Orchestrator** | the two above | bounded loop, safety guard | deterministic guard rejects bad rewrites |
| **Role Reviewer** | full enhanced resume + role profile | strengths / weaknesses / verdict | one-line "(unavailable)" on failure |
| **JD Matcher** | resume text + curated JDs | per-JD score, delta, top gaps | deterministic — never errors |

### Safety guarantees

1. **Never weakens** — guard rejects rewrites that shrink to under 50% of original.
2. **Never drops a protected term** — frameworks, models, services, numbers from the input are regex-checked.
3. **Bounded iteration** — hard cap of 4 iterations per section; per-section + job-level deadlines.
4. **Never invents** — required fields the input lacks become visible `[PLACEHOLDER]` tokens.
5. **Critic can't escape the guard** — the deterministic guard always has the final word.

### Customising the agents

Every behaviour the agents exhibit is in markdown under `skills/`. Edit any
file and reload — no code changes needed.
"""


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("RESUME_UI_PORT", "7860")),
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="slate",
            radius_size="lg",
            spacing_size="md",
        ),
    )
