"""
gradio_app.py — Clean minimal UI for the Resume Enhancer pipeline.
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

from app.core.config import settings                              # noqa: E402
from app.core.ir import PipelineResult                           # noqa: E402
from app.core.llm import (best_available_backend,               # noqa: E402
                          is_backend_configured)
from app.core.skills import load_skills                          # noqa: E402
from app.pipeline import (PipelineConfig, ROLES,                # noqa: E402
                          run_pipeline)

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)

load_skills()

_rate_limit_store: dict[str, list[float]] = {}


def _check_rate_limit(session_id: str = "default") -> Optional[str]:
    now = time.time()
    window = 3600
    max_runs = settings.max_runs_per_hour
    if session_id not in _rate_limit_store:
        _rate_limit_store[session_id] = []
    _rate_limit_store[session_id] = [t for t in _rate_limit_store[session_id] if now - t < window]
    if len(_rate_limit_store[session_id]) >= max_runs:
        remaining = int(window - (now - _rate_limit_store[session_id][0]))
        mins = max(1, remaining // 60)
        return f"Rate limit reached ({max_runs} runs/hour). Try again in ~{mins} min."
    _rate_limit_store[session_id].append(now)
    return None


def _friendly_error(raw: str) -> str:
    lower = raw.lower()
    if "api key" in lower or "unauthorized" in lower or "401" in lower:
        return "Invalid or missing API key. Check your Groq key and try again."
    if "rate limit" in lower or "429" in lower:
        return "AI service is busy. Wait a moment and try again."
    if "timeout" in lower:
        return "Request timed out. Try again or switch to a faster model."
    if "connection" in lower:
        return "Connection failed. Check your internet connection."
    return f"Something went wrong: {raw[:180]}"


# ─────────────────────────────────────────────────────────────
#  CSS — Clean minimal design
# ─────────────────────────────────────────────────────────────
CSS = """
:root {
  --bg:        #0f1117;
  --bg2:       #161b27;
  --bg3:       #1e2436;
  --border:    rgba(255,255,255,0.08);
  --border2:   rgba(255,255,255,0.14);
  --text:      #f1f5f9;
  --text2:     #94a3b8;
  --text3:     #475569;
  --accent:    #6366f1;
  --accent2:   #818cf8;
  --green:     #22c55e;
  --amber:     #f59e0b;
  --red:       #ef4444;
  --radius:    12px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, html { background: var(--bg) !important; }

.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  padding: 0 16px !important;
}

/* ── Header ── */
.app-header {
  padding: 40px 0 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}
.app-title {
  font-size: 26px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.03em;
  margin-bottom: 6px;
}
.app-title span { color: var(--accent2); }
.app-sub {
  font-size: 14px;
  color: var(--text2);
  line-height: 1.6;
}

/* ── Tabs ── */
.gradio-container .tab-nav {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 0 !important;
  padding: 0 !important;
  margin-bottom: 28px !important;
  gap: 0 !important;
}
.gradio-container .tab-nav button {
  color: var(--text3) !important;
  font-weight: 500 !important;
  font-size: 13.5px !important;
  border-radius: 0 !important;
  padding: 10px 18px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color 0.15s, border-color 0.15s !important;
  letter-spacing: 0 !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
  box-shadow: none !important;
}
.gradio-container .tab-nav button:hover:not(.selected) {
  color: var(--text2) !important;
  background: transparent !important;
}

/* ── Form inputs ── */
.gradio-container label,
.gradio-container .label-wrap span {
  color: var(--text2) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em !important;
  text-transform: uppercase !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--bg2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
  font-size: 14px !important;
  transition: border-color 0.15s !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
  outline: none !important;
}
.gradio-container .block,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .form { background: transparent !important; }
.gradio-container .prose { color: var(--text2) !important; }

/* Dropdown */
.gradio-container .svelte-select,
.gradio-container select {
  background: var(--bg2) !important;
  color: var(--text) !important;
  border-color: var(--border2) !important;
  border-radius: var(--radius) !important;
}
.gradio-container .svelte-select .listContainer {
  background: var(--bg3) !important;
  border-color: var(--border2) !important;
  border-radius: var(--radius) !important;
}
.gradio-container .svelte-select .item { color: var(--text) !important; }
.gradio-container .svelte-select .item.active { background: rgba(99,102,241,0.15) !important; }

/* Radio */
.gradio-container .radio-group {
  background: transparent !important;
  gap: 6px !important;
}
.gradio-container .radio-group label {
  text-transform: none !important;
  font-size: 13.5px !important;
  color: var(--text2) !important;
  font-weight: 400 !important;
  letter-spacing: 0 !important;
}

/* File upload */
.gradio-container .upload-button,
.gradio-container .file-preview {
  background: var(--bg2) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  color: var(--text2) !important;
}
.gradio-container .upload-button:hover {
  border-color: var(--accent) !important;
  background: rgba(99,102,241,0.06) !important;
}

/* Accordion */
.gradio-container .accordion {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.gradio-container details > summary {
  color: var(--text2) !important;
  font-size: 13px !important;
}

/* Code block */
.gradio-container .code-wrap {
  background: var(--bg2) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Button ── */
.gradio-container button.primary {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 12px 24px !important;
  cursor: pointer !important;
  transition: background 0.15s, transform 0.1s !important;
  letter-spacing: -0.01em !important;
}
.gradio-container button.primary:hover {
  background: #4f46e5 !important;
  transform: translateY(-1px) !important;
}
.gradio-container button.primary:active { transform: none !important; }

/* ── Section label ── */
.field-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text3);
  margin-bottom: 10px;
  margin-top: 20px;
}

/* ── Status pills / score badges ── */
.pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.pill-green  { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.pill-amber  { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.pill-red    { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.pill-indigo { background: rgba(99,102,241,0.12); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }
.pill-gray   { background: rgba(255,255,255,0.06); color: var(--text2); border: 1px solid var(--border2); }

/* ── KPI row ── */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.kpi-box {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
}
.kpi-val {
  font-size: 24px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.04em;
  line-height: 1;
  margin-bottom: 4px;
}
.kpi-lbl {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text3);
  font-weight: 600;
}
.kpi-note { font-size: 11.5px; color: var(--text3); margin-top: 4px; }

/* ── Score bar ── */
.score-bar { height: 3px; background: var(--border2); border-radius: 999px; overflow: hidden; margin-top: 8px; }
.score-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.bar-green  { background: var(--green); }
.bar-amber  { background: var(--amber); }
.bar-red    { background: var(--red); }
.bar-indigo { background: var(--accent2); }

/* ── Trace card ── */
.trace-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 12px;
}
.trace-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.trace-label {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text);
}
.trace-meta { display: flex; align-items: center; gap: 8px; }
.diff-block {
  background: var(--bg3);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
}
.diff-tag {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: var(--text3);
  margin-bottom: 6px;
}
.diff-text {
  font-size: 13px;
  color: var(--text2);
  line-height: 1.6;
  white-space: pre-wrap;
}
.diff-text.after { color: var(--text); }
.iter-row { display: flex; align-items: center; gap: 6px; margin-top: 10px; }
.iter-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--bg3); border: 1px solid var(--border2);
}
.iter-dot.accept { background: var(--green); border-color: var(--green); }
.iter-dot.done   { background: var(--accent); border-color: var(--accent); }
.iter-info { font-size: 11.5px; color: var(--text3); margin-left: 4px; }
.trace-note { font-size: 11.5px; color: var(--amber); margin-top: 6px; }
.violations { font-size: 11.5px; color: var(--text3); margin-top: 6px; line-height: 1.6; }

/* ── Progress ── */
.progress-wrap {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
}
.progress-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.progress-title { font-size: 13.5px; font-weight: 600; color: var(--text); }
.progress-eta   { font-size: 12px; color: var(--text3); }
.prog-track {
  height: 3px;
  background: var(--border2);
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 18px;
}
.prog-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 999px;
  transition: width 0.4s ease;
}
.stage-row {
  display: flex;
  align-items: center;
  gap: 0;
  margin-bottom: 18px;
  overflow-x: auto;
}
.stage-node { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.stage-icon {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
  background: var(--bg3); border: 1.5px solid var(--border2);
  color: var(--text3);
  transition: all 0.2s;
}
.stage-icon.active { border-color: var(--accent); color: var(--accent); background: rgba(99,102,241,0.1); }
.stage-icon.done   { border-color: var(--green);  color: var(--green);  background: rgba(34,197,94,0.1); }
.stage-lbl { font-size: 10px; color: var(--text3); margin-top: 4px; letter-spacing: 0.04em; }
.stage-lbl.active { color: var(--accent2); }
.stage-lbl.done   { color: var(--green); }
.stage-conn {
  flex: 1; height: 1.5px; background: var(--border2);
  min-width: 20px; margin-bottom: 16px;
  transition: background 0.3s;
}
.stage-conn.done { background: var(--green); }
.log-feed {
  font-size: 12px;
  color: var(--text3);
  line-height: 1.8;
  max-height: 180px;
  overflow-y: auto;
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.log-feed .lok { color: var(--green); }
.log-feed .ls  { color: var(--accent2); }
.log-feed .ld  { color: var(--text3); }

/* ── Review ── */
.review-hero {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
  margin-bottom: 16px;
}
.review-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.review-role { font-size: 17px; font-weight: 700; color: var(--text); }
.review-sub  { font-size: 12.5px; color: var(--text3); margin-top: 3px; }
.score-circle {
  text-align: center;
  flex-shrink: 0;
}
.score-num {
  font-size: 36px;
  font-weight: 800;
  letter-spacing: -0.05em;
  line-height: 1;
}
.score-num.high  { color: var(--green); }
.score-num.mid   { color: var(--amber); }
.score-num.low   { color: var(--red); }
.score-caption { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-top: 3px; }
.verdict {
  font-size: 13px;
  color: var(--text2);
  font-style: italic;
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  line-height: 1.5;
}
.review-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.review-col {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
}
.review-col h5 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); font-weight: 700; margin-bottom: 10px; }
.review-col ul { padding-left: 16px; }
.review-col li { font-size: 13px; color: var(--text2); line-height: 1.7; }
.review-col.weak li { color: #fca5a5; }
.kw-row {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
}
.kw-row h5 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); font-weight: 700; margin-bottom: 10px; }
.tag {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border2);
  color: var(--text2);
  margin: 3px;
}
.tag.miss { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.2); color: #f87171; }

/* ── JD table ── */
.jd-block { margin-bottom: 20px; }
.jd-title { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.nc-table-wrap { overflow-x: auto; }
table.nc-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.nc-table th {
  text-align: left;
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text3);
  font-weight: 700;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}
table.nc-table td {
  padding: 9px 10px;
  color: var(--text2);
  border-bottom: 1px solid var(--border);
}
table.nc-table tr:last-child td { border-bottom: none; }
table.nc-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.nc-table td.gap { color: var(--text3); font-size: 12px; }

/* ── Error / warning ── */
.msg-error {
  background: rgba(239,68,68,0.07);
  border: 1px solid rgba(239,68,68,0.2);
  border-radius: var(--radius);
  padding: 16px 18px;
  color: #fca5a5;
  font-size: 13.5px;
  line-height: 1.5;
}
.msg-warn {
  background: rgba(245,158,11,0.07);
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: var(--radius);
  padding: 12px 16px;
  color: #fde68a;
  font-size: 12.5px;
  margin-top: 12px;
}
.msg-empty {
  color: var(--text3);
  font-size: 13.5px;
  padding: 40px 20px;
  text-align: center;
  border: 1px dashed var(--border2);
  border-radius: var(--radius);
  line-height: 1.6;
}

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 20px 0; }

/* ── Footer ── */
.app-footer {
  text-align: center;
  color: var(--text3);
  font-size: 12px;
  padding: 24px 0 16px;
  border-top: 1px solid var(--border);
  margin-top: 32px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── Responsive ── */
@media (max-width: 768px) {
  .review-cols { grid-template-columns: 1fr; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
"""


# ─────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────
def _sc(score: float) -> str:
    if score >= 75: return "high"
    if score >= 50: return "mid"
    return "low"


def _bar(score: float, cls: str = "") -> str:
    pct = max(0.0, min(100.0, score))
    bar_cls = cls or f"bar-{'green' if pct >= 75 else 'amber' if pct >= 50 else 'red'}"
    return f'<div class="score-bar"><div class="score-bar-fill {bar_cls}" style="width:{pct:.0f}%"></div></div>'


def _pill(text: str, kind: str = "gray") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def _empty(msg: str) -> str:
    return f'<div class="msg-empty">{msg}</div>'


def _kpi(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi-box">'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-lbl">{label}</div>'
        f'{note_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────
#  Result panels
# ─────────────────────────────────────────────────────────────
def _summary_html(result: PipelineResult) -> str:
    if result.status == "error":
        errs = "<br>".join(result.errors) or "Unknown error."
        return f'<div class="msg-error"><b>Pipeline failed</b><br>{errs}</div>'

    sections_changed = sum(1 for t in result.section_traces if t.changed)
    sections_total   = len(result.section_traces)
    elapsed          = f"{result.elapsed_ms / 1000:.1f}s"
    ats              = result.ats.score if result.ats else 0.0
    review_score     = result.role_reviews[0].overall_score if result.role_reviews else 0.0
    jd_after         = result.jd_report.avg_score_after if result.jd_report else 0.0
    jd_delta         = result.jd_report.avg_delta if result.jd_report else 0.0
    role_pretty      = ROLES.get(result.role, result.role)
    delta_str        = f"{jd_delta:+.1f} vs before" if jd_delta else ""

    kpis = "".join([
        _kpi("Sections changed",  f"{sections_changed}/{sections_total}"),
        _kpi("ATS score",         f"{ats:.0f}",         f"/100"),
        _kpi("Hiring manager",    f"{review_score:.0f}", f"/100 · {role_pretty}"),
        _kpi("JD match",          f"{jd_after:.0f}",    delta_str),
        _kpi("Elapsed",           elapsed),
    ])

    warnings = ""
    if result.warnings:
        ws = "  ·  ".join(result.warnings[:3])
        warnings = f'<div class="msg-warn">{ws}</div>'

    return f'<div class="kpi-row">{kpis}</div>{warnings}'


def _sections_html(result: PipelineResult) -> str:
    if not result.section_traces:
        return _empty("Section traces will appear here after enhancement.")
    rows = []
    for t in result.section_traces:
        sc = _sc(t.final_score)
        changed_pill = _pill("changed", "green") if t.changed else _pill("unchanged", "gray")
        score_pill   = _pill(f"{t.final_score:.0f}", sc if sc != "high" else "indigo")

        iter_dots = ""
        if t.iterations:
            dots = []
            for s in t.iterations:
                cls = "accept" if s.accepted else ("done" if s.verdict != "error" else "")
                dots.append(f'<span class="iter-dot {cls}" title="iter {s.iteration}: {s.score:.0f}"></span>')
            iter_dots = (
                '<div class="iter-row">'
                + "".join(dots)
                + f'<span class="iter-info">{t.iterations_used} iter · score {t.final_score:.0f}/100</span>'
                + '</div>'
            )

        violations = ""
        if t.iterations and t.iterations[-1].violations:
            vlist = "; ".join(t.iterations[-1].violations[:3])
            violations = f'<div class="violations">Critic: {vlist}</div>'

        note = f'<div class="trace-note">{t.note}</div>' if t.note else ""

        rows.append(
            f'<div class="trace-card">'
            f'<div class="trace-head">'
            f'<span class="trace-label">{t.label}</span>'
            f'<div class="trace-meta">{score_pill}{changed_pill}</div>'
            f'</div>'
            f'<div class="diff-block"><div class="diff-tag">Before</div><div class="diff-text">{t.before}</div></div>'
            f'<div class="diff-block"><div class="diff-tag">After</div><div class="diff-text after">{t.after}</div></div>'
            f'{violations}{iter_dots}{note}'
            f'</div>'
        )
    return "\n".join(rows)


def _review_html(result: PipelineResult) -> str:
    if not result.role_reviews:
        return _empty("Hiring-manager review will appear here after enhancement.")
    r  = result.role_reviews[0]
    sc = _sc(r.overall_score)

    strengths  = "".join(f"<li>{s}</li>" for s in r.strengths)  or "<li><i>None returned</i></li>"
    weaknesses = "".join(f"<li>{w}</li>" for w in r.weaknesses) or "<li><i>None</i></li>"
    missing    = "".join(f'<span class="tag miss">{k}</span>' for k in r.missing_keywords) or '<span class="tag">None</span>'

    hero = (
        f'<div class="review-hero">'
        f'<div class="review-top">'
        f'<div><div class="review-role">{r.role_name}</div>'
        f'<div class="review-sub">Simulated hiring-manager evaluation</div></div>'
        f'<div class="score-circle">'
        f'<div class="score-num {sc}">{r.overall_score:.0f}</div>'
        f'<div class="score-caption">Phone-screen<br>likelihood</div>'
        f'</div></div>'
        f'{_bar(r.overall_score)}'
        f'<div class="verdict">"{r.one_line_verdict}"</div>'
        f'</div>'
    )

    cols = (
        f'<div class="review-cols">'
        f'<div class="review-col"><h5>Strengths</h5><ul>{strengths}</ul></div>'
        f'<div class="review-col weak"><h5>Weaknesses</h5><ul>{weaknesses}</ul></div>'
        f'</div>'
    )

    kw = (
        f'<div class="kw-row">'
        f'<h5>Missing keywords</h5>'
        f'{missing}'
        f'</div>'
    )

    extra = ""
    if len(result.role_reviews) > 1:
        rows = []
        for rr in result.role_reviews[1:]:
            kc = _sc(rr.overall_score)
            rows.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:10px 0;border-bottom:1px solid var(--border)">'
                f'<div><div style="font-size:13px;color:var(--text);font-weight:600">{rr.role_name}</div>'
                f'<div style="font-size:12px;color:var(--text3);margin-top:2px">"{rr.one_line_verdict}"</div></div>'
                f'{_pill(str(int(rr.overall_score)), kc if kc != "high" else "indigo")}'
                f'</div>'
            )
        extra = (
            f'<div style="background:var(--bg2);border:1px solid var(--border);'
            f'border-radius:var(--radius);padding:16px 18px;margin-top:12px">'
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;'
            f'color:var(--text3);font-weight:700;margin-bottom:10px">Cross-role scores</div>'
            + "".join(rows) + "</div>"
        )

    return hero + cols + kw + extra


def _jd_html(result: PipelineResult) -> str:
    bits = []
    if result.jd_report and result.jd_report.samples:
        bits.append(_jd_block(result.jd_report, "Target role"))
    for r in result.cross_role_jd_reports:
        bits.append(_jd_block(r, "Cross-validation"))
    if not bits:
        return _empty("JD matching is off or no JDs are loaded for this role.")
    return "\n".join(bits)


def _jd_block(report, kind: str) -> str:
    role_pretty = ROLES.get(report.role_id, report.role_id)
    delta_cls = "high" if report.avg_delta >= 5 else ("mid" if report.avg_delta >= 0 else "low")

    rows = [
        "<tr><th>Job Description</th><th>Archetype</th>"
        "<th class='num'>Before</th><th class='num'>After</th>"
        "<th class='num'>Δ</th><th>Top gaps</th></tr>"
    ]
    for s in report.samples:
        klass = "green" if s.delta >= 5 else ("amber" if s.delta >= 0 else "red")
        gaps  = ", ".join(s.missing_keywords[:4]) or "—"
        rows.append(
            f'<tr><td><b style="color:var(--text)">{s.title}</b></td>'
            f'<td><span class="tag">{s.company_archetype}</span></td>'
            f'<td class="num">{s.score_before:.0f}</td>'
            f'<td class="num">{s.score_after:.0f}</td>'
            f'<td class="num">{_pill(f"{s.delta:+.1f}", klass)}</td>'
            f'<td class="gap">{gaps}</td></tr>'
        )

    title_html = (
        f'<div class="jd-title">'
        f'<span style="color:var(--text);font-weight:700">{role_pretty}</span>'
        f'{_pill(kind, "gray")}'
        f'<span style="font-size:13px;color:var(--text2);font-weight:400;margin-left:auto">'
        f'{report.avg_score_before:.1f} → {report.avg_score_after:.1f} '
        f'{_pill(f"{report.avg_delta:+.1f}", delta_cls if delta_cls != "high" else "indigo")}'
        f'</span></div>'
    )

    gaps_html = ""
    if report.top_gaps:
        gaps_html = (
            f'<div style="margin-top:12px">'
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;'
            f'color:var(--text3);font-weight:700;margin-bottom:8px">Top gaps</div>'
            + "".join(f'<span class="tag miss">{k}</span>' for k in report.top_gaps)
            + "</div>"
        )

    return (
        f'<div class="jd-block" style="background:var(--bg2);border:1px solid var(--border);'
        f'border-radius:var(--radius);padding:18px 20px">'
        f'{title_html}'
        f'<div class="nc-table-wrap"><table class="nc-table">{"".join(rows)}</table></div>'
        f'{gaps_html}</div>'
    )


# ─────────────────────────────────────────────────────────────
#  Progress rendering
# ─────────────────────────────────────────────────────────────
_STAGE_DEFS = [
    ("parse",    "📄", "Parse"),
    ("repair",   "🔧", "Repair"),
    ("plan",     "📐", "Plan"),
    ("enhance",  "✨", "Enhance"),
    ("render",   "🖨",  "Render"),
    ("jd_match", "🎯", "Validate"),
]
_ENHANCE_SUB = {"enhance_plan", "section"}


def _stage_status(name: str, done: set, current: str) -> str:
    norm = "enhance" if name == "enhance_plan" else name
    if norm in done: return "done"
    if current and (norm == current or (norm == "enhance" and current in _ENHANCE_SUB)):
        return "active"
    return ""


def _progress_html(lines: list[str], pct: int, *, eta_s: int = 0, done: set | None = None, current: str = "") -> str:
    done = done or set()
    eta  = f"~{eta_s}s remaining" if eta_s > 0 else "finishing…"

    nodes = []
    for i, (sid, icon, label) in enumerate(_STAGE_DEFS):
        st = _stage_status(sid, done, current)
        nodes.append(
            f'<div class="stage-node">'
            f'<div class="stage-icon {st}">{icon}</div>'
            f'<div class="stage-lbl {st}">{label}</div>'
            f'</div>'
        )
        if i < len(_STAGE_DEFS) - 1:
            conn = "done" if sid in done else ""
            nodes.append(f'<div class="stage-conn {conn}"></div>')

    log_items = "".join(f"<div>{ln}</div>" for ln in lines[-24:])

    return (
        f'<div class="progress-wrap">'
        f'<div class="progress-head">'
        f'<span class="progress-title">Processing your resume</span>'
        f'<span class="progress-eta">{pct}% · {eta}</span>'
        f'</div>'
        f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
        f'<div class="stage-row">{"".join(nodes)}</div>'
        f'<div class="log-feed">{log_items}</div>'
        f'</div>'
    )


def _fmt_event(event: str, data: dict) -> str:
    if event == "stage":
        name = data.get("name", "")
        status = data.get("status", "")
        friendly = {
            "parse": "Parsing resume", "repair": "Repairing fields",
            "complete": "Filling placeholders", "plan": "Planning rewrite",
            "enhance_plan": "Enhance plan ready", "render": "Rendering LaTeX",
            "jd_match": "JD validation",
        }.get(name, "Processing")
        css = "lok" if status == "done" else "ls"
        extra = ""
        if name == "enhance_plan" and status == "done":
            extra = f' <span class="ld">({data.get("total_units", 0)} items)</span>'
        return f'<span class="{css}">{friendly} · {status}</span>{extra}'
    if event == "section":
        label  = data.get("label", "")
        status = data.get("status", "")
        if status == "done":
            return f'<span class="lok">✓</span> <span class="ld">{label}</span>'
        return f'<span class="ld">↻ {label}</span>'
    if event == "review":
        return f'<span class="ls">review · {data.get("status", "")}</span>'
    return f'<span class="ld">{event}</span>'


# ─────────────────────────────────────────────────────────────
#  Enhance handler
# ─────────────────────────────────────────────────────────────
def _enhance_handler(
    file,
    role_id: str,
    groq_api_key: str,
    optimization_mode: str,
):
    blank = (
        _empty("Section traces appear here after enhancement."),
        _empty("Hiring-manager review appears here."),
        _empty("JD scores appear here."),
        None, "",
        gr.update(visible=False),
    )

    if file is None:
        yield (_empty("Upload a <b>.tex</b> resume, choose a target role, then click <b>Enhance Resume</b>."), *blank)
        return

    tex_path = Path(file.name if hasattr(file, "name") else file)
    if tex_path.suffix.lower() != ".tex":
        yield (_empty("Please upload a <b>.tex</b> file. PDF is not supported."), *blank)
        return

    try:
        fsize = tex_path.stat().st_size
        if fsize > settings.max_upload_bytes:
            yield (_empty(f"File too large ({fsize // 1024} KB). Max: {settings.max_upload_kb} KB."), *blank)
            return
    except Exception:
        pass

    rate_err = _check_rate_limit()
    if rate_err:
        yield (_empty(rate_err), *blank)
        return

    if groq_api_key.strip():
        os.environ["GROQ_API_KEY"] = groq_api_key.strip()

    if not is_backend_configured("groq") and best_available_backend() is None:
        yield (_empty("No AI backend configured. Add a <b>Groq API key</b> above."), *blank)
        return

    section_budget = 140 if optimization_mode == "accuracy" else (100 if optimization_mode == "balanced" else 70)
    use_multi_llm  = optimization_mode in ("accuracy", "balanced")

    cfg = PipelineConfig(
        role_id=role_id,
        backend="groq" if is_backend_configured("groq") else "auto",
        enable_critic=True,
        enable_role_review=True,
        enable_jd_matching=True,
        enable_cross_role=False,
        max_iterations=4,
        enable_multi_llm=use_multi_llm,
        max_section_calls=section_budget,
        optimization_mode=optimization_mode,
    )

    q: queue.Queue = queue.Queue()
    holder: dict   = {"result": None, "error": None}

    def _progress(event: str, data: dict) -> None:
        try:
            q.put((event, data), timeout=1.0)
        except queue.Full:
            pass

    def _worker() -> None:
        try:
            holder["result"] = run_pipeline(tex_path, cfg, progress=_progress, is_file=True)
        except Exception as e:
            log.exception("[ui] pipeline crashed")
            holder["error"] = str(e)
        finally:
            q.put(("done", {}))

    t = threading.Thread(target=_worker, daemon=True, name="enhance-worker")
    t.start()

    progress_lines: list[str] = ['<span class="ls">[start]</span> initializing…']
    progress_pct   = 3
    stage_weights  = {"parse": 10, "repair": 10, "complete": 8, "plan": 8,
                      "enhance_plan": 4, "render": 15, "jd_match": 15}
    stages_done: set[str] = set()
    current_stage  = "parse"
    rewrite_done   = 0
    rewrite_total  = 0
    start_ts       = time.monotonic()
    last_emit      = time.monotonic()

    while True:
        try:
            event, data = q.get(timeout=0.6)
        except queue.Empty:
            if time.monotonic() - last_emit > 1.2:
                elapsed = max(1, int(time.monotonic() - start_ts))
                eta     = max(0, int((elapsed / max(progress_pct, 1)) * (100 - progress_pct)))
                yield (
                    _progress_html(progress_lines, progress_pct, eta_s=eta, done=stages_done, current=current_stage),
                    *blank[:-3], None, "", gr.update(visible=False),
                )
                last_emit = time.monotonic()
            continue

        if event == "done":
            break

        if event == "stage":
            name   = data.get("name", "")
            status = data.get("status", "")
            norm   = "enhance" if name in ("enhance_plan",) else name
            if status == "done":
                stages_done.add(norm)
                progress_pct = min(95, progress_pct + stage_weights.get(name, 2))
            else:
                current_stage = norm
            if name == "enhance_plan" and status == "done":
                rewrite_total = int(data.get("total_units", 0) or 0)
                progress_pct  = max(progress_pct, 35)

        if event == "section" and data.get("status") == "done":
            rewrite_done += 1
            if rewrite_total > 0:
                progress_pct = max(progress_pct, min(90, 35 + int((rewrite_done / rewrite_total) * 35)))

        progress_lines.append(_fmt_event(event, data))
        last_emit = time.monotonic()
        elapsed   = max(1, int(time.monotonic() - start_ts))
        eta       = max(0, int((elapsed / max(progress_pct, 1)) * (100 - progress_pct)))
        yield (
            _progress_html(progress_lines, progress_pct, eta_s=eta, done=stages_done, current=current_stage),
            *blank[:-3], None, "", gr.update(visible=False),
        )

    if holder["error"]:
        yield (
            f'<div class="msg-error"><b>Enhancement failed</b><br>{_friendly_error(holder["error"])}</div>',
            *blank,
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


# ─────────────────────────────────────────────────────────────
#  Build app
# ─────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    role_choices = [(name, rid) for rid, name in ROLES.items()]

    with gr.Blocks(
        title="AI Resume Enhancer",
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
            radius_size="md",
        ),
        css=CSS,
    ) as app:

        gr.HTML(
            '<div class="app-header">'
            '<div class="app-title">AI Resume Enhancer <span>·</span></div>'
            '<div class="app-sub">'
            'Upload your .tex resume, pick a target role, and get back an ATS-optimised, '
            'Overleaf-ready .tex — facts preserved, no hallucinations.'
            '</div></div>'
        )

        with gr.Tabs():

            # ── Enhance ──────────────────────────────────────────
            with gr.Tab("Enhance"):
                with gr.Row():
                    # Left column — controls
                    with gr.Column(scale=1, min_width=280):

                        gr.HTML('<div class="field-label">Resume file</div>')
                        file_in = gr.File(
                            label="Upload .tex",
                            file_types=[".tex"],
                            file_count="single",
                            height=110,
                        )

                        gr.HTML('<div class="field-label">Target role</div>')
                        role_in = gr.Dropdown(
                            choices=role_choices,
                            value="ai_ml_engineer",
                            label="Role",
                            interactive=True,
                        )

                        gr.HTML('<div class="field-label">Quality mode</div>')
                        opt_mode_in = gr.Radio(
                            choices=[
                                ("Accuracy", "accuracy"),
                                ("Balanced", "balanced"),
                                ("Speed", "speed"),
                            ],
                            value="accuracy",
                            label="Mode",
                        )

                        gr.HTML(
                            '<div class="field-label">Groq API key</div>'
                            '<div style="font-size:12px;color:var(--text3);margin-bottom:8px">'
                            'Free at <a href="https://console.groq.com" target="_blank" '
                            'style="color:var(--accent2)">console.groq.com</a> — no credit card'
                            '</div>'
                        )
                        groq_key_in = gr.Textbox(
                            label="Groq API key",
                            type="password",
                            placeholder="gsk_...",
                            show_label=False,
                        )

                        run_btn = gr.Button(
                            "Enhance Resume",
                            variant="primary",
                            size="lg",
                        )

                    # Right column — live output
                    with gr.Column(scale=2):
                        summary_out = gr.HTML(
                            _empty("Upload a .tex resume and click <b>Enhance Resume</b> to begin.")
                        )

            # ── Sections ─────────────────────────────────────────
            with gr.Tab("Sections"):
                sections_out = gr.HTML(
                    _empty("Per-section before / after will appear here after enhancement.")
                )

            # ── Review ───────────────────────────────────────────
            with gr.Tab("Review"):
                review_out = gr.HTML(
                    _empty("Hiring-manager review will appear here after enhancement.")
                )

            # ── JD Matching ───────────────────────────────────────
            with gr.Tab("JD Match"):
                jd_out = gr.HTML(
                    _empty("JD keyword scores will appear here after enhancement.")
                )

            # ── Download ─────────────────────────────────────────
            with gr.Tab("Download"):
                with gr.Group(visible=False) as download_group:
                    gr.HTML(
                        '<div style="font-size:13px;color:var(--text2);margin-bottom:16px">'
                        'Paste the .tex into <a href="https://overleaf.com" target="_blank" '
                        'style="color:var(--accent2)">Overleaf</a> to compile to PDF.'
                        '</div>'
                    )
                    tex_file_out = gr.File(label="Enhanced .tex", interactive=False)
                    tex_text_out = gr.Code(
                        label="Preview",
                        language="latex",
                        lines=30,
                        interactive=False,
                    )

        # Wire up
        run_btn.click(
            _enhance_handler,
            inputs=[file_in, role_in, groq_key_in, opt_mode_in],
            outputs=[
                summary_out, sections_out, review_out, jd_out,
                tex_file_out, tex_text_out, download_group,
            ],
        )

        gr.HTML(
            '<div class="app-footer">'
            'AI Resume Enhancer · Multi-agent · Fact-preserving · Overleaf-ready'
            '</div>'
        )

    return app


if __name__ == "__main__":
    app = build_app()
    auth = None
    if settings.auth_enabled:
        auth = (settings.auth_user, settings.auth_pass)
        log.info("[ui] basic auth enabled for user: %s", settings.auth_user)
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("RESUME_UI_PORT", "7860")),
        show_error=True,
        auth=auth,
    )
