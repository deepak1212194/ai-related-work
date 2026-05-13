"""
gradio_app.py — AI Resume Enhancer UI (v3).

Layout
------
Left sidebar  : Upload · Role · JD mode toggle (Generic / Custom) · Run button
                + Run History panel (last 5 runs)
Right area    : Progress panel (while running)
                Results tabs after completion:
                  1. Summary      — KPI cards + persistent-gaps callout
                  2. Action Plan  — manual checklist + gap classification
                  3. Sections     — before/after diffs with critic detail
                  4. JD Match     — generic tab OR custom JD tab (toggled)
                  5. HM Review    — hiring-manager simulation
                  6. Download     — .tex file + source preview
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

from app.core.config import settings                               # noqa: E402
from app.core.history import get_persistent_gaps, load_runs       # noqa: E402
from app.core.ir import PipelineResult                            # noqa: E402
from app.core.skills import load_skills                           # noqa: E402
from app.pipeline import PipelineConfig, ROLES, run_pipeline      # noqa: E402

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
        return "Invalid or missing API key. Check your configuration."
    if "rate limit" in lower or "429" in lower:
        return "AI service is busy. Wait a moment and try again."
    if "timeout" in lower:
        return "Request timed out. Try again shortly."
    if "connection" in lower:
        return "Connection failed. Check your internet connection."
    return f"Something went wrong: {raw[:200]}"


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg:         #09090b;
  --bg1:        #111115;
  --bg2:        #18181c;
  --bg3:        #1f1f26;
  --bg4:        #27272f;
  --border:     rgba(255,255,255,0.06);
  --border2:    rgba(255,255,255,0.11);
  --border3:    rgba(255,255,255,0.18);
  --text:       #fafafa;
  --text2:      #a1a1aa;
  --text3:      #52525b;
  --text4:      #3f3f46;
  --violet:     #8b5cf6;
  --violet2:    #a78bfa;
  --violet3:    #c4b5fd;
  --cyan:       #06b6d4;
  --emerald:    #10b981;
  --amber:      #f59e0b;
  --rose:       #f43f5e;
  --blue:       #3b82f6;
  --radius:     12px;
  --radius-sm:  8px;
  --radius-lg:  18px;
  --sidebar-w:  300px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, html {
  background: var(--bg) !important;
  min-height: 100vh;
}

.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  padding: 0 16px 60px !important;
}

footer { display: none !important; }
.gradio-container > .built-with { display: none !important; }

.gradio-container .block,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .form,
.gradio-container .gap,
.gradio-container .padded {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* ── Compact hero ── */
.hero {
  padding: 36px 0 28px;
  text-align: center;
  position: relative;
}
.hero-glow {
  position: absolute;
  top: 0; left: 50%;
  transform: translateX(-50%);
  width: 500px; height: 160px;
  background: radial-gradient(ellipse, rgba(139,92,246,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(139,92,246,0.1);
  border: 1px solid rgba(139,92,246,0.22);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 600;
  color: var(--violet3);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 16px;
}
.hero-badge::before {
  content: '';
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--violet2);
  box-shadow: 0 0 5px var(--violet);
}
.hero-title {
  font-size: clamp(26px, 4vw, 40px);
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.04em;
  line-height: 1.1;
  margin-bottom: 10px;
}
.hero-title em {
  font-style: normal;
  background: linear-gradient(135deg, var(--violet2) 0%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 14px;
  color: var(--text2);
  max-width: 440px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ── Layout: sidebar + main ── */
.app-layout {
  display: grid;
  grid-template-columns: var(--sidebar-w) 1fr;
  gap: 16px;
  align-items: start;
}
@media (max-width: 860px) {
  .app-layout { grid-template-columns: 1fr; }
}

/* ── Sidebar ── */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  position: sticky;
  top: 16px;
}
.card {
  background: var(--bg1);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  overflow: hidden;
}
.card-hd {
  padding: 14px 16px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-hd-icon {
  width: 26px; height: 26px;
  border-radius: 7px;
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.18);
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
}
.card-hd-title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}
.card-body { padding: 16px; }

/* ── JD mode toggle ── */
.jd-toggle {
  display: flex;
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: var(--radius-sm);
  padding: 3px;
  gap: 3px;
  margin-bottom: 10px;
}
.jd-toggle-btn {
  flex: 1;
  padding: 7px 0;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  font-family: 'Inter', system-ui, sans-serif;
  transition: background 0.15s, color 0.15s;
  background: transparent;
  color: var(--text3);
}
.jd-toggle-btn.active {
  background: var(--violet);
  color: #fff;
}

/* ── Run button ── */
.gradio-container button.primary {
  background: linear-gradient(135deg, var(--violet) 0%, #6d28d9 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 13px 20px !important;
  cursor: pointer !important;
  letter-spacing: -0.01em !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  transition: opacity 0.15s, transform 0.1s, box-shadow 0.15s !important;
  width: 100% !important;
  margin-top: 4px !important;
}
.gradio-container button.primary:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(139,92,246,0.35) !important;
}
.gradio-container button.primary:active {
  transform: none !important;
}

/* ── Form inputs ── */
.gradio-container label > span,
.gradio-container .label-wrap span {
  color: var(--text2) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 13.5px !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  transition: border-color 0.15s !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
  outline: none !important;
}
.gradio-container .prose { color: var(--text2) !important; }

.gradio-container .svelte-select {
  background: var(--bg3) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
}
.gradio-container .svelte-select .listContainer {
  background: var(--bg4) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
}
.gradio-container .svelte-select .item { color: var(--text) !important; font-size: 13px !important; }
.gradio-container .svelte-select .item.active,
.gradio-container .svelte-select .item:hover {
  background: rgba(139,92,246,0.12) !important;
}

.gradio-container .upload-button,
.gradio-container .file-preview {
  background: var(--bg3) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  color: var(--text3) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  transition: border-color 0.15s !important;
}
.gradio-container .upload-button:hover {
  border-color: var(--violet) !important;
  background: rgba(139,92,246,0.04) !important;
}

.gradio-container .code-wrap,
.gradio-container code {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Fira Code', 'SF Mono', monospace !important;
}

/* ── Tabs ── */
.gradio-container .tab-nav {
  background: var(--bg2) !important;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 0 !important;
  padding: 0 20px !important;
  margin: 0 !important;
  gap: 0 !important;
}
.gradio-container .tab-nav button {
  color: var(--text3) !important;
  font-weight: 500 !important;
  font-size: 12.5px !important;
  border-radius: 0 !important;
  padding: 12px 14px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color 0.15s !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--violet) !important;
}
.gradio-container .tab-nav button:hover:not(.selected) {
  color: var(--text2) !important;
}
.gradio-container .tabitem {
  background: transparent !important;
  padding: 0 !important;
}

/* ── KPI grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 8px;
  margin-bottom: 16px;
}
.kpi-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1.5px;
  background: linear-gradient(90deg, transparent, var(--violet), transparent);
  opacity: 0.4;
}
.kpi-num {
  font-size: 24px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.05em;
  line-height: 1;
  margin-bottom: 4px;
  font-family: 'Inter', system-ui, sans-serif;
}
.kpi-key {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text3);
  font-weight: 600;
  font-family: 'Inter', system-ui, sans-serif;
}
.kpi-note { font-size: 10.5px; color: var(--text4); margin-top: 2px; font-family: 'Inter', system-ui, sans-serif; }

/* ── Score bar ── */
.score-bar {
  height: 2px;
  background: var(--border2);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 8px;
}
.score-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.bar-g { background: linear-gradient(90deg, var(--emerald), #6ee7b7); }
.bar-a { background: var(--amber); }
.bar-r { background: var(--rose); }
.bar-v { background: linear-gradient(90deg, var(--violet), var(--violet2)); }

/* ── Pills ── */
.pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.01em;
  font-family: 'Inter', system-ui, sans-serif;
}
.pill-v  { background: rgba(139,92,246,0.12); color: var(--violet3); border: 1px solid rgba(139,92,246,0.2); }
.pill-c  { background: rgba(6,182,212,0.10);  color: #67e8f9;        border: 1px solid rgba(6,182,212,0.2); }
.pill-g  { background: rgba(16,185,129,0.10); color: #6ee7b7;        border: 1px solid rgba(16,185,129,0.2); }
.pill-a  { background: rgba(245,158,11,0.10); color: #fcd34d;        border: 1px solid rgba(245,158,11,0.2); }
.pill-r  { background: rgba(244,63,94,0.10);  color: #fda4af;        border: 1px solid rgba(244,63,94,0.2); }
.pill-z  { background: rgba(255,255,255,0.05); color: var(--text2);  border: 1px solid var(--border2); }
.pill-b  { background: rgba(59,130,246,0.10); color: #93c5fd;        border: 1px solid rgba(59,130,246,0.2); }

/* ── Section trace ── */
.trace-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 8px;
  transition: border-color 0.15s;
}
.trace-card:hover { border-color: var(--border2); }
.trace-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.trace-name { font-size: 13px; font-weight: 600; color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
.trace-meta { display: flex; align-items: center; gap: 5px; }
.diff-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.diff-pane { background: var(--bg3); border-radius: var(--radius-sm); padding: 11px 13px; }
.diff-lbl {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  color: var(--text4);
  margin-bottom: 5px;
  font-family: 'Inter', system-ui, sans-serif;
}
.diff-txt {
  font-size: 12px;
  color: var(--text2);
  line-height: 1.65;
  white-space: pre-wrap;
  font-family: 'Inter', system-ui, sans-serif;
}
.diff-txt.new { color: var(--text); }
.iter-row { display: flex; align-items: center; gap: 4px; margin-top: 9px; }
.iter-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--bg4); border: 1px solid var(--border2); }
.iter-dot.ok   { background: var(--emerald); border-color: var(--emerald); }
.iter-dot.done { background: var(--violet);  border-color: var(--violet); }
.iter-meta { font-size: 10.5px; color: var(--text3); margin-left: 3px; font-family: 'Inter', system-ui, sans-serif; }
.trace-warn { font-size: 11px; color: var(--amber); margin-top: 5px; font-family: 'Inter', system-ui, sans-serif; }
.violations { font-size: 11px; color: var(--text3); margin-top: 4px; line-height: 1.5; font-family: 'Inter', system-ui, sans-serif; }

/* ── Progress panel ── */
.prog-panel {
  background: var(--bg1);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  padding: 20px 22px;
}
.prog-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.prog-title { font-size: 13.5px; font-weight: 600; color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
.prog-pct { font-size: 12px; color: var(--text3); font-variant-numeric: tabular-nums; font-family: 'Inter', system-ui, sans-serif; }
.prog-track { height: 2px; background: var(--border2); border-radius: 999px; overflow: hidden; margin-bottom: 20px; }
.prog-fill { height: 100%; background: linear-gradient(90deg, var(--violet), var(--cyan)); border-radius: 999px; transition: width 0.4s ease; }
.stage-pipeline { display: flex; align-items: center; gap: 0; margin-bottom: 18px; overflow-x: auto; }
.stage-node { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.stage-icon {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
  background: var(--bg3);
  border: 1.5px solid var(--border2);
  color: var(--text3);
  transition: all 0.2s;
}
.stage-icon.active { border-color: var(--violet); color: var(--violet2); background: rgba(139,92,246,0.1); box-shadow: 0 0 8px rgba(139,92,246,0.2); }
.stage-icon.done   { border-color: var(--emerald); color: var(--emerald); background: rgba(16,185,129,0.08); }
.stage-lbl { font-size: 9px; color: var(--text4); margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600; font-family: 'Inter', system-ui, sans-serif; }
.stage-lbl.active { color: var(--violet3); }
.stage-lbl.done   { color: var(--emerald); }
.stage-conn { flex: 1; height: 1px; background: var(--border2); min-width: 14px; margin-bottom: 16px; transition: background 0.3s; }
.stage-conn.done { background: var(--emerald); }
.log-stream { font-size: 11px; color: var(--text3); line-height: 1.85; max-height: 140px; overflow-y: auto; font-family: 'Fira Code', 'SF Mono', monospace; }
.log-stream .ok  { color: var(--emerald); }
.log-stream .act { color: var(--violet2); }
.log-stream .dim { color: var(--text4); }

/* ── Action plan ── */
.action-section { margin-bottom: 20px; }
.action-section-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-weight: 700;
  color: var(--text3);
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
  font-family: 'Inter', system-ui, sans-serif;
}
.action-item {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--amber);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 12px 14px;
  margin-bottom: 7px;
}
.action-item.urgent { border-left-color: var(--rose); }
.action-item.ok { border-left-color: var(--emerald); }
.action-item.info { border-left-color: var(--blue); }
.action-section-label {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 3px;
  font-family: 'Inter', system-ui, sans-serif;
}
.action-issue { font-size: 11.5px; color: var(--text2); margin-bottom: 4px; font-family: 'Inter', system-ui, sans-serif; }
.action-hint { font-size: 11px; color: var(--text3); font-style: italic; font-family: 'Inter', system-ui, sans-serif; }
.gap-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 600px) { .gap-grid { grid-template-columns: 1fr; } }
.gap-col { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 14px; }
.gap-col-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; margin-bottom: 8px; font-family: 'Inter', system-ui, sans-serif; }
.gap-col-title.presentation { color: var(--amber); }
.gap-col-title.real { color: var(--rose); }
.gap-tag {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 5px;
  font-size: 11.5px;
  font-weight: 500;
  margin: 3px 2px;
  font-family: 'Inter', system-ui, sans-serif;
}
.gap-tag.presentation { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); color: #fcd34d; }
.gap-tag.real { background: rgba(244,63,94,0.08); border: 1px solid rgba(244,63,94,0.2); color: #fda4af; }
.persistent-gap-banner {
  background: rgba(59,130,246,0.07);
  border: 1px solid rgba(59,130,246,0.18);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  margin-bottom: 14px;
  font-size: 12px;
  color: #93c5fd;
  line-height: 1.55;
  font-family: 'Inter', system-ui, sans-serif;
}
.persistent-gap-banner strong { color: #bfdbfe; }

/* ── Review panel ── */
.review-hero {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.review-left { flex: 1; min-width: 0; }
.review-role-name { font-size: 17px; font-weight: 700; color: var(--text); letter-spacing: -0.02em; margin-bottom: 3px; font-family: 'Inter', system-ui, sans-serif; }
.review-sub { font-size: 11.5px; color: var(--text3); margin-bottom: 12px; font-family: 'Inter', system-ui, sans-serif; }
.verdict { font-size: 12.5px; color: var(--text2); font-style: italic; padding-top: 12px; margin-top: 12px; border-top: 1px solid var(--border); line-height: 1.55; font-family: 'Inter', system-ui, sans-serif; }
.score-ring { flex-shrink: 0; text-align: center; width: 72px; }
.score-big { font-size: 38px; font-weight: 800; letter-spacing: -0.06em; line-height: 1; font-family: 'Inter', system-ui, sans-serif; }
.score-big.hi  { color: var(--emerald); }
.score-big.mid { color: var(--amber); }
.score-big.lo  { color: var(--rose); }
.score-cap { font-size: 8.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text3); margin-top: 2px; font-weight: 600; font-family: 'Inter', system-ui, sans-serif; }
.review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.review-col { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 14px; }
.review-col h5 { font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); font-weight: 700; margin-bottom: 8px; font-family: 'Inter', system-ui, sans-serif; }
.review-col ul { padding-left: 15px; }
.review-col li { font-size: 12px; color: var(--text2); line-height: 1.7; font-family: 'Inter', system-ui, sans-serif; }
.review-col.weak li { color: #fda4af; }
.kw-panel { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px 14px; }
.kw-panel h5 { font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); font-weight: 700; margin-bottom: 8px; font-family: 'Inter', system-ui, sans-serif; }
.kw-tag {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border2);
  color: var(--text2);
  margin: 2px;
  font-family: 'Inter', system-ui, sans-serif;
}
.kw-tag.miss { background: rgba(244,63,94,0.07); border-color: rgba(244,63,94,0.18); color: #fda4af; }

/* ── JD table ── */
.jd-section { margin-bottom: 14px; }
.jd-header { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 10px; display: flex; align-items: center; gap: 7px; font-family: 'Inter', system-ui, sans-serif; }
.jd-table-wrap { overflow-x: auto; }
table.jd-table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: 'Inter', system-ui, sans-serif; }
table.jd-table th { text-align: left; font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text3); font-weight: 700; padding: 7px 9px; border-bottom: 1px solid var(--border); }
table.jd-table td { padding: 8px 9px; color: var(--text2); border-bottom: 1px solid var(--border); }
table.jd-table tr:last-child td { border-bottom: none; }
table.jd-table td.r { text-align: right; font-variant-numeric: tabular-nums; }
table.jd-table td.dim { color: var(--text3); font-size: 11px; }
table.jd-table tbody tr:hover td { background: rgba(255,255,255,0.02); }

/* ── History sidebar panel ── */
.hist-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
  gap: 8px;
}
.hist-item:last-child { border-bottom: none; }
.hist-left { flex: 1; min-width: 0; }
.hist-role { font-size: 11.5px; font-weight: 600; color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
.hist-meta { font-size: 10px; color: var(--text3); margin-top: 2px; font-family: 'Inter', system-ui, sans-serif; }
.hist-scores { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }

/* ── Messages ── */
.msg-error {
  background: rgba(244,63,94,0.06);
  border: 1px solid rgba(244,63,94,0.18);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  color: #fda4af;
  font-size: 13px;
  line-height: 1.55;
  font-family: 'Inter', system-ui, sans-serif;
}
.msg-warn {
  background: rgba(245,158,11,0.06);
  border: 1px solid rgba(245,158,11,0.18);
  border-radius: var(--radius-sm);
  padding: 9px 13px;
  color: #fcd34d;
  font-size: 11.5px;
  margin-top: 8px;
  font-family: 'Inter', system-ui, sans-serif;
}
.msg-idle {
  color: var(--text4);
  font-size: 13px;
  padding: 42px 20px;
  text-align: center;
  border: 1px dashed var(--border2);
  border-radius: var(--radius);
  line-height: 1.6;
  font-family: 'Inter', system-ui, sans-serif;
}
.msg-idle strong { color: var(--text3); }

/* ── Download tab ── */
.dl-hint { font-size: 12.5px; color: var(--text2); margin-bottom: 14px; line-height: 1.5; font-family: 'Inter', system-ui, sans-serif; }
.dl-hint a { color: var(--violet2); text-decoration: none; }
.dl-hint a:hover { text-decoration: underline; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--border3); }

/* ── Responsive ── */
@media (max-width: 720px) {
  .diff-wrap { grid-template-columns: 1fr; }
  .review-hero { flex-direction: column; }
  .review-grid { grid-template-columns: 1fr; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .gap-grid { grid-template-columns: 1fr; }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sc(s: float) -> str:
    return "hi" if s >= 75 else ("mid" if s >= 50 else "lo")


def _bar(s: float) -> str:
    p = max(0, min(100, s))
    c = "bar-g" if p >= 75 else ("bar-a" if p >= 50 else "bar-r")
    return f'<div class="score-bar"><div class="score-bar-fill {c}" style="width:{p:.0f}%"></div></div>'


def _pill(t: str, k: str = "z") -> str:
    return f'<span class="pill pill-{k}">{t}</span>'


def _idle(msg: str) -> str:
    return f'<div class="msg-idle">{msg}</div>'


def _kpi(label: str, val: str, note: str = "") -> str:
    n = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-num">{val}</div>'
        f'<div class="kpi-key">{label}</div>'
        f'{n}</div>'
    )


def _ts_ago(ts: float) -> str:
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff/60)}m ago"
    if diff < 86400:
        return f"{int(diff/3600)}h ago"
    return f"{int(diff/86400)}d ago"


# ─────────────────────────────────────────────────────────────────────────────
#  Result renderers
# ─────────────────────────────────────────────────────────────────────────────
def _score_badge(val: float, label: str = "") -> str:
    """Circular score badge with colour coding."""
    color = "var(--emerald)" if val >= 75 else ("var(--amber)" if val >= 50 else "var(--rose)")
    sub = f'<div style="font-size:9px;color:var(--text3);margin-top:3px;font-weight:600;letter-spacing:.05em">{label}</div>' if label else ""
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'background:var(--bg3);border:1.5px solid {color}33;border-radius:10px;'
        f'padding:10px 14px;min-width:64px;text-align:center">'
        f'<div style="font-size:26px;font-weight:800;color:{color};letter-spacing:-0.05em;line-height:1">'
        f'{val:.0f}</div>'
        f'{sub}'
        f'</div>'
    )


def _delta_badge(delta: float) -> str:
    if delta > 0:
        color, icon = "var(--emerald)", "▲"
    elif delta < 0:
        color, icon = "var(--rose)", "▼"
    else:
        color, icon = "var(--text3)", "–"
    return (
        f'<span style="font-size:13px;font-weight:700;color:{color};'
        f'letter-spacing:-0.02em">{icon} {abs(delta):.1f}</span>'
    )


def _section_divider(title: str, icon: str = "") -> str:
    ico = f'<span style="font-size:13px;margin-right:6px">{icon}</span>' if icon else ""
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:20px 0 10px">'
        f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;'
        f'font-weight:700;color:var(--text3)">{ico}{title}</div>'
        f'<div style="flex:1;height:1px;background:var(--border)"></div>'
        f'</div>'
    )


def _summary_html(r: PipelineResult) -> str:
    if r.status == "error":
        errs = "<br>".join(r.errors) or "Unknown error."
        return f'<div class="msg-error"><b>Pipeline failed</b><br>{errs}</div>'

    changed = sum(1 for t in r.section_traces if t.changed)
    total   = len(r.section_traces)
    elapsed = f"{r.elapsed_ms / 1000:.1f}s"
    ats     = r.ats.score if r.ats else 0.0
    rev     = r.role_reviews[0].overall_score if r.role_reviews else 0.0
    jd_r    = r.custom_jd_report or r.jd_report
    jd_a    = jd_r.avg_score_after if jd_r else 0.0
    jd_d    = jd_r.avg_delta if jd_r else 0.0
    role    = ROLES.get(r.role, r.role)
    jd_lbl  = "Custom JD" if r.custom_jd_report else "JD match"

    # ── Score scoreboard row ──
    scores_row = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">'
        f'{_score_badge(ats, "ATS")}'
        f'{_score_badge(rev, "HM Score")}'
        f'{_score_badge(jd_a, jd_lbl)}'
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'background:var(--bg3);border:1.5px solid var(--border2);border-radius:10px;'
        f'padding:10px 14px;min-width:64px;text-align:center">'
        f'<div style="font-size:26px;font-weight:800;line-height:1">{_delta_badge(jd_d)}</div>'
        f'<div style="font-size:9px;color:var(--text3);margin-top:3px;font-weight:600;letter-spacing:.05em">JD delta</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'background:var(--bg3);border:1.5px solid var(--border2);border-radius:10px;'
        f'padding:10px 14px;min-width:64px;text-align:center">'
        f'<div style="font-size:26px;font-weight:800;color:var(--violet2);letter-spacing:-.05em;line-height:1">'
        f'{changed}<span style="font-size:14px;color:var(--text3)">/{total}</span></div>'
        f'<div style="font-size:9px;color:var(--text3);margin-top:3px;font-weight:600;letter-spacing:.05em">Sections</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'background:var(--bg3);border:1.5px solid var(--border2);border-radius:10px;'
        f'padding:10px 14px;min-width:64px;text-align:center">'
        f'<div style="font-size:26px;font-weight:800;color:var(--text2);letter-spacing:-.05em;line-height:1">'
        f'{elapsed}</div>'
        f'<div style="font-size:9px;color:var(--text3);margin-top:3px;font-weight:600;letter-spacing:.05em">Time</div>'
        f'</div>'
        f'</div>'
    )

    # ── ATS breakdown ──
    ats_html = ""
    if r.ats:
        ats_html = (
            f'<div style="background:var(--bg2);border:1px solid var(--border);'
            f'border-radius:var(--radius-sm);padding:14px;margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
            f'<div style="font-size:12.5px;font-weight:600;color:var(--text)">ATS Keyword Coverage</div>'
            f'<div style="font-size:12px;color:var(--text3)">{r.ats.matched_count} / {r.ats.total_checked} matched</div>'
            f'</div>'
            f'{_bar(ats)}'
            f'<div style="display:flex;gap:16px;margin-top:10px;font-size:11.5px;color:var(--text3)">'
            f'<span style="color:var(--emerald)">✓ {r.ats.matched_count} matched</span>'
            f'<span style="color:var(--rose)">✗ {len(r.ats.missing_high_impact)} missing</span>'
            f'<span style="color:var(--amber)">⚠ {len(r.ats.presentation_gaps)} presentation gaps</span>'
            f'</div>'
            f'</div>'
        )

    # ── Manual actions callout ──
    actions_html = ""
    n_actions = len(r.manual_actions)
    if n_actions > 0:
        actions_html = (
            f'<div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.18);'
            f'border-radius:var(--radius-sm);padding:12px 14px;margin-bottom:10px">'
            f'<div style="font-size:12px;color:#fcd34d;font-weight:600">'
            f'⚠ {n_actions} section{"s" if n_actions > 1 else ""} need manual attention</div>'
            f'<div style="font-size:11.5px;color:var(--text3);margin-top:4px">'
            f'See the Action Plan tab for specific fixes.</div>'
            f'</div>'
        )

    # ── Persistent gaps callout ──
    pg = get_persistent_gaps(8)
    pg_html = ""
    if pg:
        tags = " ".join(f'<span class="kw-tag miss">{k}</span>' for k in pg)
        pg_html = (
            f'<div class="persistent-gap-banner">'
            f'<strong>Recurring gaps</strong> — these keywords have been missing '
            f'across multiple runs. Consider addressing them in your resume:<br>'
            f'<div style="margin-top:7px">{tags}</div>'
            f'</div>'
        )

    warn = ""
    if r.warnings:
        warn = f'<div class="msg-warn">{"  ·  ".join(r.warnings[:3])}</div>'

    # ── Role label ──
    role_tag = (
        f'<div style="font-size:11px;color:var(--text3);margin-bottom:14px">'
        f'Target role: <span style="color:var(--violet3);font-weight:600">{role}</span>'
        f'</div>'
    )

    return role_tag + scores_row + ats_html + actions_html + pg_html + warn


def _action_plan_html(r: PipelineResult) -> str:
    """Tab 2: Manual action checklist + gap classification."""
    parts = []

    # ── Manual actions ──
    parts.append(_section_divider("Manual fixes needed", "🔧"))
    if r.manual_actions:
        items_html = []
        for a in r.manual_actions:
            urgency = "urgent" if a.score < 50 else ("ok" if a.score >= 75 else "")
            sc_color = "var(--rose)" if a.score < 50 else ("var(--amber)" if a.score < 75 else "var(--emerald)")
            items_html.append(
                f'<div class="action-item {urgency}" style="margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px">'
                f'<div class="action-section-label">{a.section}</div>'
                f'<span style="font-size:11px;font-weight:700;color:{sc_color};'
                f'background:{sc_color}18;padding:2px 7px;border-radius:4px;white-space:nowrap">'
                f'score {a.score:.0f}</span>'
                f'</div>'
                f'<div class="action-issue">⚑ {a.issue}</div>'
                f'<div class="action-hint" style="margin-top:4px;padding-top:4px;'
                f'border-top:1px solid var(--border)">→ {a.fix_hint}</div>'
                f'</div>'
            )
        parts.append("".join(items_html))
    else:
        parts.append(
            '<div class="action-item ok">'
            '<div class="action-issue" style="color:var(--emerald)">✓ All sections met the quality threshold — no residual issues.</div>'
            '</div>'
        )

    # ── Gap classification ──
    if r.ats:
        pres = r.ats.presentation_gaps
        real = r.ats.real_gaps
        parts.append(_section_divider("Keyword gap analysis", "🎯"))
        pres_html = (
            "".join(f'<span class="gap-tag presentation">{k}</span>' for k in pres)
            or '<span style="color:var(--text4);font-size:12px">None — great!</span>'
        )
        real_html = (
            "".join(f'<span class="gap-tag real">{k}</span>' for k in real)
            or '<span style="color:var(--text4);font-size:12px">None — great!</span>'
        )
        parts.append(
            '<div class="gap-grid">'
            '<div class="gap-col">'
            '<div class="gap-col-title presentation">⚡ Presentation gaps</div>'
            '<div style="font-size:11px;color:var(--text3);line-height:1.5;margin-bottom:10px">'
            'These keywords exist somewhere in your resume but aren\'t visible enough to ATS scanners. '
            'Move them to the Skills section or lead bullets.</div>'
            f'{pres_html}'
            '</div>'
            '<div class="gap-col">'
            '<div class="gap-col-title real">🚨 Real skill gaps</div>'
            '<div style="font-size:11px;color:var(--text3);line-height:1.5;margin-bottom:10px">'
            'These keywords are completely absent from your resume. '
            'Add them through a certification, project, or new bullet where truthful.</div>'
            f'{real_html}'
            '</div>'
            '</div>'
        )

    # ── ATS matched keywords ──
    if r.ats and r.ats.matched:
        parts.append(_section_divider("Keywords already matched", "✓"))
        matched_html = " ".join(f'<span class="kw-tag">{k}</span>' for k in r.ats.matched[:30])
        parts.append(f'<div class="kw-panel">{matched_html}</div>')

    if not parts:
        return _idle("Action plan appears after enhancement.")
    return "\n".join(parts)


def _sections_html(r: PipelineResult) -> str:
    if not r.section_traces:
        return _idle("Section details appear here after enhancement.")

    changed_count   = sum(1 for t in r.section_traces if t.changed)
    avg_score       = (sum(t.final_score for t in r.section_traces) / len(r.section_traces)) if r.section_traces else 0

    # Summary strip
    summary_strip = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;'
        f'background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);'
        f'padding:12px 16px;margin-bottom:14px;align-items:center">'
        f'<div style="font-size:12px;color:var(--text2)">'
        f'<span style="color:var(--emerald);font-weight:600">{changed_count}</span> / '
        f'{len(r.section_traces)} sections rewritten · '
        f'avg critic score <span style="color:var(--violet2);font-weight:600">{avg_score:.0f}</span>/100'
        f'</div>'
        f'</div>'
    )

    out = [summary_strip]
    for t in r.section_traces:
        sc_color = "var(--emerald)" if t.final_score >= 75 else ("var(--amber)" if t.final_score >= 50 else "var(--rose)")
        cp = _pill("rewritten", "g") if t.changed else _pill("unchanged", "z")

        # Dim scores breakdown
        dim_html = ""
        if t.iterations and t.iterations[-1].dim_scores:
            dims = t.iterations[-1].dim_scores
            dim_items = "".join(
                f'<div style="text-align:center;min-width:52px">'
                f'<div style="font-size:14px;font-weight:700;color:{"var(--emerald)" if v >= 16 else ("var(--amber)" if v >= 10 else "var(--rose)")}">'
                f'{int(v)}</div>'
                f'<div style="font-size:9px;color:var(--text4);text-transform:uppercase;letter-spacing:.05em;margin-top:1px">'
                f'{k.replace("_"," ")[:10]}</div>'
                f'</div>'
                for k, v in dims.items()
            )
            dim_html = (
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;padding:10px 0 2px;'
                f'border-top:1px solid var(--border);margin-top:10px">'
                f'{dim_items}'
                f'</div>'
            )

        dots = ""
        if t.iterations:
            d = "".join(
                f'<span class="iter-dot {"ok" if s.accepted else ("done" if s.verdict != "error" else "")}" '
                f'title="iter {s.iteration}: {s.score:.0f}"></span>'
                for s in t.iterations
            )
            dots = (
                f'<div class="iter-row" style="margin-top:8px">{d}'
                f'<span class="iter-meta">{t.iterations_used} iteration{"s" if t.iterations_used > 1 else ""}</span>'
                f'</div>'
            )

        viols = ""
        if t.iterations and t.iterations[-1].violations:
            vlist = "".join(f'<div style="margin-top:3px">• {v}</div>' for v in t.iterations[-1].violations[:3])
            viols = (
                f'<div style="background:rgba(245,158,11,0.05);border-left:2px solid var(--amber);'
                f'padding:8px 10px;margin-top:8px;border-radius:0 4px 4px 0;'
                f'font-size:11px;color:var(--amber);line-height:1.55">'
                f'{vlist}</div>'
            )

        note = ""
        if t.note:
            note = (
                f'<div style="font-size:11px;color:var(--rose);margin-top:6px;'
                f'padding:6px 10px;background:rgba(244,63,94,0.05);border-radius:4px">'
                f'⚠ {t.note}</div>'
            )

        out.append(
            f'<div class="trace-card">'
            # header
            f'<div class="trace-head">'
            f'<div style="display:flex;flex-direction:column;gap:3px;flex:1;min-width:0">'
            f'<span class="trace-name">{t.label}</span>'
            f'</div>'
            f'<div class="trace-meta" style="gap:6px;flex-shrink:0">'
            f'<span style="font-size:20px;font-weight:800;color:{sc_color};letter-spacing:-.04em">'
            f'{t.final_score:.0f}</span>'
            f'<span style="font-size:10px;color:var(--text4)">/100</span>'
            f'{cp}'
            f'</div>'
            f'</div>'
            # diff
            f'<div class="diff-wrap">'
            f'<div class="diff-pane">'
            f'<div class="diff-lbl">Before</div>'
            f'<div class="diff-txt">{t.before}</div>'
            f'</div>'
            f'<div class="diff-pane">'
            f'<div class="diff-lbl">After</div>'
            f'<div class="diff-txt new">{t.after}</div>'
            f'</div>'
            f'</div>'
            f'{dim_html}{dots}{viols}{note}'
            f'</div>'
        )
    return "\n".join(out)


def _review_html(r: PipelineResult) -> str:
    if not r.role_reviews:
        return _idle("Hiring-manager review appears after enhancement.")
    rv = r.role_reviews[0]
    sc = _sc(rv.overall_score)

    strengths  = "".join(f"<li style='margin-bottom:4px'>{s}</li>" for s in rv.strengths)  or "<li><i>None</i></li>"
    weaknesses = "".join(f"<li style='margin-bottom:4px'>{w}</li>" for w in rv.weaknesses) or "<li><i>None</i></li>"
    missing    = "".join(f'<span class="kw-tag miss">{k}</span>' for k in rv.missing_keywords) or '<span class="kw-tag">None</span>'

    hero = (
        f'<div class="review-hero">'
        f'<div class="review-left">'
        f'<div class="review-role-name">{rv.role_name}</div>'
        f'<div class="review-sub">Simulated hiring-manager evaluation</div>'
        f'{_bar(rv.overall_score)}'
        f'<div class="verdict">"{rv.one_line_verdict}"</div>'
        f'</div>'
        f'<div class="score-ring">'
        f'<div class="score-big {sc}">{rv.overall_score:.0f}</div>'
        f'<div class="score-cap">Phone-screen<br>likelihood</div>'
        f'</div></div>'
    )
    cols = (
        f'<div class="review-grid">'
        f'<div class="review-col">'
        f'<h5>✓ Strengths</h5>'
        f'<ul style="list-style:none;padding:0">{strengths}</ul>'
        f'</div>'
        f'<div class="review-col weak">'
        f'<h5>✗ Weaknesses</h5>'
        f'<ul style="list-style:none;padding:0">{weaknesses}</ul>'
        f'</div>'
        f'</div>'
    )
    kw = (
        f'<div class="kw-panel">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
        f'<h5 style="margin:0">Missing role keywords</h5>'
        f'<span style="font-size:11px;color:var(--text3)">{len(rv.missing_keywords)} keywords</span>'
        f'</div>'
        f'{missing}'
        f'</div>'
    )

    extra = ""
    if len(r.role_reviews) > 1:
        rows = []
        for rr in r.role_reviews[1:]:
            kc = "v" if rr.overall_score >= 75 else ("a" if rr.overall_score >= 50 else "r")
            rows.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 0;border-bottom:1px solid var(--border)">'
                f'<div><div style="font-size:12.5px;color:var(--text);font-weight:600">{rr.role_name}</div>'
                f'<div style="font-size:11px;color:var(--text3);margin-top:1px">"{rr.one_line_verdict}"</div></div>'
                f'{_pill(str(int(rr.overall_score)), kc)}'
                f'</div>'
            )
        extra = (
            f'<div style="background:var(--bg2);border:1px solid var(--border);'
            f'border-radius:var(--radius-sm);padding:12px 14px;margin-top:8px">'
            f'<div style="font-size:9.5px;text-transform:uppercase;letter-spacing:.08em;'
            f'color:var(--text3);font-weight:700;margin-bottom:7px">Cross-role scores</div>'
            + "".join(rows) + "</div>"
        )
    return hero + cols + kw + extra


def _jd_html(r: PipelineResult) -> str:
    """Render JD match tab: custom JD first if present, then generic."""
    bits = []
    if r.custom_jd_report and r.custom_jd_report.samples_count > 0:
        bits.append(_jd_block(r.custom_jd_report, "Your Job Description", is_custom=True))
    if r.jd_report and r.jd_report.samples:
        bits.append(_jd_block(r.jd_report, "Generic role samples"))
    for jd in r.cross_role_jd_reports:
        bits.append(_jd_block(jd, "Cross-validation"))
    if not bits:
        return _idle("JD matching scores appear here after enhancement.")
    return "\n".join(bits)


def _jd_block(rep, kind: str, *, is_custom: bool = False) -> str:
    role = "Custom JD" if is_custom else ROLES.get(rep.role_id, rep.role_id)
    dk   = "g" if rep.avg_delta >= 5 else ("a" if rep.avg_delta >= 0 else "r")
    avg_color = "var(--emerald)" if rep.avg_score_after >= 75 else ("var(--amber)" if rep.avg_score_after >= 50 else "var(--rose)")

    # ── Header with score ring ──
    hdr = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;gap:12px">'
        f'<div>'
        f'<div style="font-size:14px;font-weight:700;color:var(--text);letter-spacing:-.02em">{role}</div>'
        f'<div style="margin-top:5px">{_pill(kind, "b" if is_custom else "z")}</div>'
        f'</div>'
        f'<div style="display:flex;gap:10px;align-items:center;flex-shrink:0">'
        f'<div style="text-align:center">'
        f'<div style="font-size:11px;color:var(--text3);margin-bottom:3px">Before</div>'
        f'<div style="font-size:22px;font-weight:800;color:var(--text2);letter-spacing:-.04em">'
        f'{rep.avg_score_before:.0f}</div>'
        f'</div>'
        f'<div style="font-size:18px;color:var(--text4)">→</div>'
        f'<div style="text-align:center">'
        f'<div style="font-size:11px;color:var(--text3);margin-bottom:3px">After</div>'
        f'<div style="font-size:22px;font-weight:800;color:{avg_color};letter-spacing:-.04em">'
        f'{rep.avg_score_after:.0f}</div>'
        f'</div>'
        f'<div style="text-align:center;padding:8px 12px;border-radius:8px;'
        f'background:var(--bg3);border:1px solid var(--border2)">'
        f'<div style="font-size:10px;color:var(--text4);margin-bottom:2px">Δ avg</div>'
        f'{_delta_badge(rep.avg_delta)}'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    # ── Table ──
    rows = [
        "<thead><tr>"
        "<th>Job Title</th>"
        "<th>Archetype</th>"
        "<th class='r'>Before</th>"
        "<th class='r'>After</th>"
        "<th class='r'>Δ</th>"
        "<th>Missing keywords</th>"
        "</tr></thead>"
        "<tbody>"
    ]
    for s in rep.samples:
        kl      = "g" if s.delta > 3 else ("a" if s.delta >= 0 else "r")
        delta_c = "var(--emerald)" if s.delta > 3 else ("var(--amber)" if s.delta >= 0 else "var(--rose)")
        after_c = "var(--emerald)" if s.score_after >= 75 else ("var(--amber)" if s.score_after >= 50 else "var(--text2)")
        # Display up to 5 missing keywords as small tags
        gap_tags = "".join(
            f'<span style="display:inline-block;background:rgba(244,63,94,0.08);'
            f'border:1px solid rgba(244,63,94,0.18);color:#fda4af;'
            f'font-size:10.5px;padding:1px 6px;border-radius:4px;margin:2px">{k}</span>'
            for k in s.missing_keywords[:5]
        ) or '<span style="color:var(--text4);font-size:11px">—</span>'
        rows.append(
            f'<tr>'
            f'<td><b style="color:var(--text);font-size:12.5px">{s.title}</b>'
            f'<div style="font-size:10.5px;color:var(--text3);margin-top:1px">{s.seniority or ""}</div>'
            f'</td>'
            f'<td>{_pill(s.company_archetype or "—", "z")}</td>'
            f'<td class="r" style="font-size:13px;color:var(--text3)">{s.score_before:.0f}</td>'
            f'<td class="r" style="font-size:13px;font-weight:700;color:{after_c}">{s.score_after:.0f}</td>'
            f'<td class="r"><span style="font-size:12.5px;font-weight:700;color:{delta_c}">'
            f'{"+" if s.delta > 0 else ""}{s.delta:.1f}'
            f'</span></td>'
            f'<td style="max-width:220px">{gap_tags}</td>'
            f'</tr>'
        )
    rows.append("</tbody>")

    gaps_html = ""
    if rep.top_gaps:
        gaps_html = (
            f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">'
            f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.07em;'
            f'color:var(--text3);font-weight:700;margin-bottom:7px">Top gaps across all samples</div>'
            + "".join(
                f'<span class="kw-tag miss" style="margin:2px">{k}</span>'
                for k in rep.top_gaps
            )
            + "</div>"
        )

    return (
        f'<div class="jd-section" style="background:var(--bg2);border:1px solid var(--border);'
        f'border-radius:var(--radius);padding:18px 20px;margin-bottom:14px">'
        f'{hdr}'
        f'<div class="jd-table-wrap"><table class="jd-table">{"".join(rows)}</table></div>'
        f'{gaps_html}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  History panel
# ─────────────────────────────────────────────────────────────────────────────
def _history_html() -> str:
    runs = load_runs()[:5]
    if not runs:
        return '<div style="color:var(--text4);font-size:12px;padding:8px 0">No runs yet.</div>'
    items = []
    for run in runs:
        role_name = ROLES.get(run.get("role", ""), run.get("role", "Unknown"))
        ats  = run.get("ats_score", 0)
        hm   = run.get("hm_score", 0)
        jd   = run.get("jd_avg_after", 0)
        ts   = run.get("timestamp", 0)
        cjd  = " · custom JD" if run.get("custom_jd_used") else ""
        ago  = _ts_ago(ts) if ts else ""
        ats_pill = _pill(f"ATS {ats:.0f}", "g" if ats >= 75 else ("a" if ats >= 50 else "r"))
        hm_pill  = _pill(f"HM {hm:.0f}", "v" if hm >= 75 else ("a" if hm >= 50 else "r"))
        items.append(
            f'<div class="hist-item">'
            f'<div class="hist-left">'
            f'<div class="hist-role">{role_name}</div>'
            f'<div class="hist-meta">{ago}{cjd}</div>'
            f'</div>'
            f'<div class="hist-scores">{ats_pill}{hm_pill}</div>'
            f'</div>'
        )
    return "".join(items)


# ─────────────────────────────────────────────────────────────────────────────
#  Progress
# ─────────────────────────────────────────────────────────────────────────────
_STAGES = [
    ("parse",    "📄", "Parse"),
    ("repair",   "🔧", "Repair"),
    ("plan",     "📐", "Plan"),
    ("enhance",  "✦",  "Enhance"),
    ("render",   "🖨",  "Render"),
    ("jd_match", "🎯", "Score"),
]
_ENHANCE_SUB = {"enhance_plan", "section"}


def _stage_st(sid: str, done: set, cur: str) -> str:
    n = "enhance" if sid == "enhance_plan" else sid
    if n in done: return "done"
    if cur and (n == cur or (n == "enhance" and cur in _ENHANCE_SUB)): return "active"
    return ""


def _progress_html(lines: list[str], pct: int, *, eta_s: int = 0, done: set | None = None, cur: str = "") -> str:
    done = done or set()
    eta  = f"~{eta_s}s left" if eta_s > 0 else "finishing…"

    nodes = []
    for i, (sid, icon, lbl) in enumerate(_STAGES):
        st = _stage_st(sid, done, cur)
        nodes.append(
            f'<div class="stage-node">'
            f'<div class="stage-icon {st}">{icon}</div>'
            f'<div class="stage-lbl {st}">{lbl}</div>'
            f'</div>'
        )
        if i < len(_STAGES) - 1:
            conn = "done" if sid in done else ""
            nodes.append(f'<div class="stage-conn {conn}"></div>')

    log_html = "".join(f"<div>{ln}</div>" for ln in lines[-18:])

    return (
        f'<div class="prog-panel">'
        f'<div class="prog-top">'
        f'<span class="prog-title">Enhancing your resume…</span>'
        f'<span class="prog-pct">{pct}% · {eta}</span>'
        f'</div>'
        f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
        f'<div class="stage-pipeline">{"".join(nodes)}</div>'
        f'<div class="log-stream">{log_html}</div>'
        f'</div>'
    )


def _fmt_event(event: str, data: dict) -> str:
    if event == "stage":
        name, status = data.get("name", ""), data.get("status", "")
        labels = {
            "parse": "Parsing resume", "repair": "Repairing fields",
            "complete": "Filling gaps", "plan": "Planning rewrite",
            "enhance_plan": "Plan ready", "render": "Rendering LaTeX",
            "jd_match": "JD matching",
        }
        css = "ok" if status == "done" else "act"
        extra = ""
        if name == "enhance_plan" and status == "done":
            extra = f' <span class="dim">({data.get("total_units", 0)} blocks)</span>'
        return f'<span class="{css}">{labels.get(name, name)} · {status}</span>{extra}'
    if event == "section":
        label, status = data.get("label", ""), data.get("status", "")
        if status == "done":
            return f'<span class="ok">✓</span> <span class="dim">{label}</span>'
        return f'<span class="dim">↻ {label}</span>'
    if event == "review":
        return f'<span class="act">review · {data.get("status", "")}</span>'
    return f'<span class="dim">{event}</span>'


# ─────────────────────────────────────────────────────────────────────────────
#  Handler
# ─────────────────────────────────────────────────────────────────────────────
def _enhance_handler(file, role_id: str, custom_jd: str):
    """Main pipeline handler. Yields 8 outputs:
    summary, action_plan, sections, review, jd, tex_file, tex_text, dl_group
    """
    blank = (
        _idle("Action plan appears here."),
        _idle("Section details appear here."),
        _idle("Hiring-manager review appears here."),
        _idle("JD match scores appear here."),
        None, "",
        gr.update(visible=False),
    )

    if file is None:
        yield (_idle("Upload a <b>.tex</b> resume, select a role, then click <strong>✦ Enhance Resume</strong>."), *blank)
        return

    tex_path = Path(file.name if hasattr(file, "name") else file)
    if tex_path.suffix.lower() != ".tex":
        yield (_idle("Please upload a <b>.tex</b> file. PDF is not supported yet."), *blank)
        return

    try:
        fsize = tex_path.stat().st_size
        if fsize > settings.max_upload_bytes:
            yield (_idle(f"File too large ({fsize // 1024} KB). Max {settings.max_upload_kb} KB."), *blank)
            return
    except Exception:
        pass

    rate_err = _check_rate_limit()
    if rate_err:
        yield (_idle(rate_err), *blank)
        return

    cfg = PipelineConfig(
        role_id=role_id,
        backend="auto",
        enable_critic=settings.critic_enabled,
        enable_role_review=settings.role_review_enabled,
        enable_jd_matching=settings.jd_match_enabled,
        enable_cross_role=False,
        max_iterations=settings.max_iterations,
        enable_multi_llm=settings.enable_multi_llm,
        max_section_calls=settings.max_section_calls,
        custom_jd_text=custom_jd.strip() if custom_jd else "",
    )

    q: queue.Queue = queue.Queue()
    holder: dict   = {"result": None, "error": None}

    def _cb(event: str, data: dict) -> None:
        try:
            q.put((event, data), timeout=1.0)
        except queue.Full:
            pass

    def _worker() -> None:
        try:
            holder["result"] = run_pipeline(tex_path, cfg, progress=_cb, is_file=True)
        except Exception as e:
            log.exception("[ui] pipeline error")
            holder["error"] = str(e)
        finally:
            q.put(("done", {}))

    threading.Thread(target=_worker, daemon=True, name="enhance-worker").start()

    lines        = ['<span class="act">[start]</span> <span class="dim">initializing…</span>']
    pct          = 3
    weights      = {"parse": 10, "repair": 10, "complete": 8, "plan": 8,
                    "enhance_plan": 4, "render": 15, "jd_match": 15}
    stages_done  : set[str] = set()
    cur_stage    = "parse"
    rew_done     = 0
    rew_total    = 0
    start_ts     = time.monotonic()
    last_emit    = time.monotonic()

    while True:
        try:
            event, data = q.get(timeout=0.55)
        except queue.Empty:
            if time.monotonic() - last_emit > 1.1:
                elapsed = max(1, int(time.monotonic() - start_ts))
                eta     = max(0, int((elapsed / max(pct, 1)) * (100 - pct)))
                yield (
                    _progress_html(lines, pct, eta_s=eta, done=stages_done, cur=cur_stage),
                    *blank[:-3], None, "", gr.update(visible=False),
                )
                last_emit = time.monotonic()
            continue

        if event == "done":
            break

        if event == "stage":
            name, status = data.get("name", ""), data.get("status", "")
            norm = "enhance" if name == "enhance_plan" else name
            if status == "done":
                stages_done.add(norm)
                pct = min(95, pct + weights.get(name, 2))
            else:
                cur_stage = norm
            if name == "enhance_plan" and status == "done":
                rew_total = int(data.get("total_units", 0) or 0)
                pct = max(pct, 35)

        if event == "section" and data.get("status") == "done":
            rew_done += 1
            if rew_total > 0:
                pct = max(pct, min(90, 35 + int((rew_done / rew_total) * 35)))

        lines.append(_fmt_event(event, data))
        last_emit = time.monotonic()
        elapsed = max(1, int(time.monotonic() - start_ts))
        eta = max(0, int((elapsed / max(pct, 1)) * (100 - pct)))
        yield (
            _progress_html(lines, pct, eta_s=eta, done=stages_done, cur=cur_stage),
            *blank[:-3], None, "", gr.update(visible=False),
        )

    if holder["error"]:
        yield (
            f'<div class="msg-error"><b>Enhancement failed</b><br>{_friendly_error(holder["error"])}</div>',
            *blank,
        )
        return

    res: PipelineResult = holder["result"]
    yield (
        _summary_html(res),
        _action_plan_html(res),
        _sections_html(res),
        _review_html(res),
        _jd_html(res),
        res.tex_path,
        res.tex_content,
        gr.update(visible=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Build app
# ─────────────────────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    role_choices = [(name, rid) for rid, name in ROLES.items()]

    with gr.Blocks(
        title="AI Resume Enhancer",
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="zinc",
            radius_size="md",
        ),
        css=CSS,
    ) as app:

        # ── Hero ──
        gr.HTML(
            '<div class="hero">'
            '<div class="hero-glow"></div>'
            '<div class="hero-badge">Multi-agent · ATS-optimised · Fact-preserving</div>'
            '<div class="hero-title">Your resume,<br><em>engineered to land.</em></div>'
            '<div class="hero-sub">'
            'Upload a LaTeX resume, paste a job description (or use generic role templates), '
            'and get a fully optimised, Overleaf-ready .tex file.'
            '</div>'
            '</div>'
        )

        # ── App layout ──
        gr.HTML('<div class="app-layout">')

        # ── LEFT SIDEBAR ──
        gr.HTML('<div class="sidebar">')

        # — Input card —
        gr.HTML(
            '<div class="card">'
            '<div class="card-hd">'
            '<div class="card-hd-icon">✦</div>'
            '<div class="card-hd-title">Enhance Resume</div>'
            '</div>'
            '<div class="card-body">'
        )

        file_in = gr.File(
            label="Resume (.tex)",
            file_types=[".tex"],
            file_count="single",
            height=110,
        )
        role_in = gr.Dropdown(
            choices=role_choices,
            value="ai_ml_engineer",
            label="Target role",
            interactive=True,
        )

        # JD mode: Generic vs Custom — shown as explanatory labels
        gr.HTML(
            '<div style="margin:12px 0 6px">'
            '<div style="font-size:11px;font-weight:600;color:var(--text2);'
            'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px">'
            'Job Description Mode</div>'
            '<div style="font-size:11.5px;color:var(--text3);line-height:1.55;margin-bottom:6px">'
            'Leave blank to use built-in role templates (Generic).<br>'
            'Paste a real JD below to target that specific job (Custom).'
            '</div>'
            '</div>'
        )

        custom_jd_in = gr.Textbox(
            label="Paste job description (optional)",
            placeholder="Paste the full JD text here to target this specific job…",
            lines=6,
            max_lines=14,
        )

        run_btn = gr.Button("✦  Enhance Resume", variant="primary", size="lg")

        gr.HTML('</div></div>')  # close card-body, card

        # — Run history card —
        gr.HTML(
            '<div class="card">'
            '<div class="card-hd">'
            '<div class="card-hd-icon">🕐</div>'
            '<div class="card-hd-title">Recent runs</div>'
            '</div>'
            '<div class="card-body" style="padding:12px 16px">'
        )
        history_out = gr.HTML(_history_html())
        gr.HTML('</div></div>')  # close card-body, card

        gr.HTML('</div>')  # close .sidebar

        # ── RIGHT MAIN AREA ──
        gr.HTML('<div style="min-width:0">')

        # Summary (always visible, above tabs)
        summary_out = gr.HTML(
            _idle("Upload a <b>.tex</b> resume, select a role, then click <strong>✦ Enhance Resume</strong>.")
        )

        # Results tabs
        with gr.Tabs():
            with gr.Tab("Action Plan"):
                action_out = gr.HTML(_idle("Manual action checklist and gap analysis appear here."))

            with gr.Tab("Sections"):
                sections_out = gr.HTML(_idle("Per-section before / after diffs appear here."))

            with gr.Tab("HM Review"):
                review_out = gr.HTML(_idle("Hiring-manager review appears here."))

            with gr.Tab("JD Match"):
                jd_out = gr.HTML(_idle("JD keyword alignment scores appear here."))

            with gr.Tab("Download"):
                with gr.Group(visible=False) as dl_group:
                    gr.HTML(
                        '<div class="dl-hint" style="padding:16px 0 0">'
                        'Paste the .tex into <a href="https://overleaf.com" target="_blank">Overleaf</a> '
                        'to compile to PDF. Every bullet, skill, and section is fully editable.'
                        '</div>'
                    )
                    tex_file_out = gr.File(label="Enhanced .tex", interactive=False)
                    tex_text_out = gr.Code(label="Source preview", language="latex", lines=28, interactive=False)

        gr.HTML('</div>')  # close right main
        gr.HTML('</div>')  # close .app-layout

        # ── Wire ──
        run_btn.click(
            _enhance_handler,
            inputs=[file_in, role_in, custom_jd_in],
            outputs=[
                summary_out,
                action_out,
                sections_out,
                review_out,
                jd_out,
                tex_file_out,
                tex_text_out,
                dl_group,
            ],
        )

        # Refresh history after each run
        run_btn.click(
            lambda: _history_html(),
            inputs=[],
            outputs=[history_out],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    auth = (settings.auth_user, settings.auth_pass) if settings.auth_enabled else None
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("RESUME_UI_PORT", "7860")),
        show_error=True,
        auth=auth,
    )
