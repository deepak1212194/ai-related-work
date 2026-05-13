"""
gradio_app.py — Sleek minimal UI for the AI Resume Enhancer.
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
from app.core.skills import load_skills                          # noqa: E402
from app.pipeline import (PipelineConfig, ROLES, run_pipeline)  # noqa: E402

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


# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
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
  --green:      #22d3ee;
  --emerald:    #10b981;
  --amber:      #f59e0b;
  --rose:       #f43f5e;
  --radius:     14px;
  --radius-sm:  8px;
  --radius-lg:  20px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, html {
  background: var(--bg) !important;
  min-height: 100vh;
}

.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  padding: 0 20px 60px !important;
}

/* ── Hide Gradio footer ── */
footer { display: none !important; }
.gradio-container > .built-with { display: none !important; }

/* ── All gradio panels transparent ── */
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

/* ── Header ── */
.hero {
  padding: 56px 0 48px;
  text-align: center;
  position: relative;
}
.hero-glow {
  position: absolute;
  top: 0; left: 50%;
  transform: translateX(-50%);
  width: 600px; height: 200px;
  background: radial-gradient(ellipse, rgba(139,92,246,0.18) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(139,92,246,0.1);
  border: 1px solid rgba(139,92,246,0.25);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--violet3);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 22px;
}
.hero-badge::before {
  content: '';
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--violet2);
  box-shadow: 0 0 6px var(--violet);
}
.hero-title {
  font-size: clamp(32px, 5vw, 52px);
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.04em;
  line-height: 1.1;
  margin-bottom: 16px;
}
.hero-title em {
  font-style: normal;
  background: linear-gradient(135deg, var(--violet2) 0%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 16px;
  color: var(--text2);
  max-width: 480px;
  margin: 0 auto;
  line-height: 1.65;
  font-weight: 400;
}

/* ── Main card ── */
.main-card {
  background: var(--bg1);
  border: 1px solid var(--border2);
  border-radius: var(--radius-lg);
  overflow: hidden;
  margin-bottom: 20px;
}
.card-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}
.card-header-icon {
  width: 30px; height: 30px;
  border-radius: 8px;
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
}
.card-header-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}
.card-header-sub {
  font-size: 12px;
  color: var(--text3);
  margin-left: auto;
}
.card-body { padding: 24px; }

/* ── Tabs ── */
.gradio-container .tab-nav {
  background: var(--bg2) !important;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 0 !important;
  padding: 0 24px !important;
  margin: 0 !important;
  gap: 0 !important;
}
.gradio-container .tab-nav button {
  color: var(--text3) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  border-radius: 0 !important;
  padding: 13px 16px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color 0.15s !important;
  letter-spacing: 0 !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--violet) !important;
  background: transparent !important;
  box-shadow: none !important;
}
.gradio-container .tab-nav button:hover:not(.selected) {
  color: var(--text2) !important;
  background: transparent !important;
}
.gradio-container .tabitem {
  background: transparent !important;
  padding: 0 !important;
}

/* ── Form inputs ── */
.gradio-container label > span,
.gradio-container .label-wrap span {
  color: var(--text2) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.03em !important;
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
  font-size: 14px !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.12) !important;
  outline: none !important;
}
.gradio-container .prose { color: var(--text2) !important; }

/* Dropdown */
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
.gradio-container .svelte-select .item { color: var(--text) !important; font-size: 13.5px !important; }
.gradio-container .svelte-select .item.active,
.gradio-container .svelte-select .item:hover {
  background: rgba(139,92,246,0.12) !important;
}

/* File upload */
.gradio-container .upload-button,
.gradio-container .file-preview {
  background: var(--bg3) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  color: var(--text3) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  transition: border-color 0.15s, background 0.15s !important;
}
.gradio-container .upload-button:hover {
  border-color: var(--violet) !important;
  background: rgba(139,92,246,0.05) !important;
  color: var(--text2) !important;
}

/* Code block */
.gradio-container .code-wrap,
.gradio-container code {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Fira Code', 'SF Mono', monospace !important;
}

/* ── Button ── */
.gradio-container button.primary {
  background: var(--violet) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 13px 28px !important;
  cursor: pointer !important;
  letter-spacing: -0.01em !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  transition: opacity 0.15s, transform 0.1s, box-shadow 0.15s !important;
  box-shadow: 0 0 0 0 rgba(139,92,246,0) !important;
  width: 100% !important;
}
.gradio-container button.primary:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(139,92,246,0.3) !important;
}
.gradio-container button.primary:active {
  transform: none !important;
  box-shadow: none !important;
}

/* ── Stat pills ── */
.pill {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 6px;
  font-size: 11.5px;
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

/* ── Score bar ── */
.score-bar {
  height: 2px;
  background: var(--border2);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 10px;
}
.score-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.bar-g { background: linear-gradient(90deg, var(--emerald), var(--green)); }
.bar-a { background: var(--amber); }
.bar-r { background: var(--rose); }
.bar-v { background: linear-gradient(90deg, var(--violet), var(--violet2)); }

/* ── KPI grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
  margin-bottom: 20px;
}
.kpi-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px;
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
  font-size: 26px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.05em;
  line-height: 1;
  margin-bottom: 5px;
  font-family: 'Inter', system-ui, sans-serif;
}
.kpi-key {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text3);
  font-weight: 600;
  font-family: 'Inter', system-ui, sans-serif;
}
.kpi-note { font-size: 11px; color: var(--text4); margin-top: 3px; font-family: 'Inter', system-ui, sans-serif; }

/* ── Section trace card ── */
.trace-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 10px;
  transition: border-color 0.15s;
}
.trace-card:hover { border-color: var(--border2); }
.trace-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.trace-name {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
  font-family: 'Inter', system-ui, sans-serif;
}
.trace-meta { display: flex; align-items: center; gap: 6px; }
.diff-wrap {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.diff-pane {
  background: var(--bg3);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
}
.diff-lbl {
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  color: var(--text4);
  margin-bottom: 6px;
  font-family: 'Inter', system-ui, sans-serif;
}
.diff-txt {
  font-size: 12.5px;
  color: var(--text2);
  line-height: 1.65;
  white-space: pre-wrap;
  font-family: 'Inter', system-ui, sans-serif;
}
.diff-txt.new { color: var(--text); }
.iter-row { display: flex; align-items: center; gap: 5px; margin-top: 10px; }
.iter-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--bg4); border: 1px solid var(--border2); }
.iter-dot.ok   { background: var(--emerald); border-color: var(--emerald); }
.iter-dot.done { background: var(--violet); border-color: var(--violet); }
.iter-meta { font-size: 11px; color: var(--text3); margin-left: 4px; font-family: 'Inter', system-ui, sans-serif; }
.trace-warn { font-size: 11px; color: var(--amber); margin-top: 6px; font-family: 'Inter', system-ui, sans-serif; }
.violations { font-size: 11px; color: var(--text3); margin-top: 5px; line-height: 1.5; font-family: 'Inter', system-ui, sans-serif; }

/* ── Progress panel ── */
.prog-panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
}
.prog-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.prog-title { font-size: 14px; font-weight: 600; color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
.prog-pct { font-size: 12px; color: var(--text3); font-variant-numeric: tabular-nums; font-family: 'Inter', system-ui, sans-serif; }
.prog-track {
  height: 2px;
  background: var(--border2);
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 22px;
}
.prog-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--violet), var(--cyan));
  border-radius: 999px;
  transition: width 0.4s ease;
}
.stage-pipeline {
  display: flex;
  align-items: center;
  gap: 0;
  margin-bottom: 20px;
  overflow-x: auto;
}
.stage-node { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.stage-icon {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px;
  background: var(--bg3);
  border: 1.5px solid var(--border2);
  color: var(--text3);
  transition: all 0.2s;
}
.stage-icon.active {
  border-color: var(--violet);
  color: var(--violet2);
  background: rgba(139,92,246,0.1);
  box-shadow: 0 0 10px rgba(139,92,246,0.2);
}
.stage-icon.done {
  border-color: var(--emerald);
  color: var(--emerald);
  background: rgba(16,185,129,0.08);
}
.stage-lbl {
  font-size: 9.5px;
  color: var(--text4);
  margin-top: 5px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 600;
  font-family: 'Inter', system-ui, sans-serif;
}
.stage-lbl.active { color: var(--violet3); }
.stage-lbl.done   { color: var(--emerald); }
.stage-conn {
  flex: 1;
  height: 1px;
  background: var(--border2);
  min-width: 16px;
  margin-bottom: 18px;
  transition: background 0.3s;
}
.stage-conn.done { background: var(--emerald); }
.log-stream {
  font-size: 11.5px;
  color: var(--text3);
  line-height: 1.85;
  max-height: 160px;
  overflow-y: auto;
  font-family: 'Fira Code', 'SF Mono', monospace;
}
.log-stream .ok  { color: var(--emerald); }
.log-stream .act { color: var(--violet2); }
.log-stream .dim { color: var(--text4); }

/* ── Review panel ── */
.review-hero {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
}
.review-left { flex: 1; min-width: 0; }
.review-role-name {
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.02em;
  margin-bottom: 4px;
  font-family: 'Inter', system-ui, sans-serif;
}
.review-sub {
  font-size: 12px;
  color: var(--text3);
  margin-bottom: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.verdict {
  font-size: 13px;
  color: var(--text2);
  font-style: italic;
  padding-top: 14px;
  margin-top: 14px;
  border-top: 1px solid var(--border);
  line-height: 1.55;
  font-family: 'Inter', system-ui, sans-serif;
}
.score-ring {
  flex-shrink: 0;
  text-align: center;
  width: 80px;
}
.score-big {
  font-size: 42px;
  font-weight: 800;
  letter-spacing: -0.06em;
  line-height: 1;
  font-family: 'Inter', system-ui, sans-serif;
}
.score-big.hi  { color: var(--emerald); }
.score-big.mid { color: var(--amber); }
.score-big.lo  { color: var(--rose); }
.score-cap {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text3);
  margin-top: 3px;
  font-weight: 600;
  font-family: 'Inter', system-ui, sans-serif;
}
.review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
.review-col {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px;
}
.review-col h5 {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text3);
  font-weight: 700;
  margin-bottom: 10px;
  font-family: 'Inter', system-ui, sans-serif;
}
.review-col ul { padding-left: 16px; }
.review-col li { font-size: 12.5px; color: var(--text2); line-height: 1.7; font-family: 'Inter', system-ui, sans-serif; }
.review-col.weak li { color: #fda4af; }
.kw-panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
}
.kw-panel h5 {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text3);
  font-weight: 700;
  margin-bottom: 10px;
  font-family: 'Inter', system-ui, sans-serif;
}
.kw-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 5px;
  font-size: 11.5px;
  font-weight: 500;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border2);
  color: var(--text2);
  margin: 3px;
  font-family: 'Inter', system-ui, sans-serif;
}
.kw-tag.miss { background: rgba(244,63,94,0.07); border-color: rgba(244,63,94,0.18); color: #fda4af; }

/* ── JD table ── */
.jd-section { margin-bottom: 16px; }
.jd-header {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'Inter', system-ui, sans-serif;
}
.jd-table-wrap { overflow-x: auto; }
table.jd-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  font-family: 'Inter', system-ui, sans-serif;
}
table.jd-table th {
  text-align: left;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text3);
  font-weight: 700;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}
table.jd-table td {
  padding: 9px 10px;
  color: var(--text2);
  border-bottom: 1px solid var(--border);
}
table.jd-table tr:last-child td { border-bottom: none; }
table.jd-table td.r { text-align: right; font-variant-numeric: tabular-nums; }
table.jd-table td.dim { color: var(--text3); font-size: 11.5px; }
table.jd-table tbody tr:hover td { background: rgba(255,255,255,0.02); }

/* ── Download tab ── */
.dl-hint {
  font-size: 13px;
  color: var(--text2);
  margin-bottom: 16px;
  line-height: 1.5;
  font-family: 'Inter', system-ui, sans-serif;
}
.dl-hint a { color: var(--violet2); text-decoration: none; }
.dl-hint a:hover { text-decoration: underline; }

/* ── Messages ── */
.msg-error {
  background: rgba(244,63,94,0.06);
  border: 1px solid rgba(244,63,94,0.18);
  border-radius: var(--radius-sm);
  padding: 16px 18px;
  color: #fda4af;
  font-size: 13.5px;
  line-height: 1.55;
  font-family: 'Inter', system-ui, sans-serif;
}
.msg-warn {
  background: rgba(245,158,11,0.06);
  border: 1px solid rgba(245,158,11,0.18);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  color: #fcd34d;
  font-size: 12px;
  margin-top: 10px;
  font-family: 'Inter', system-ui, sans-serif;
}
.msg-idle {
  color: var(--text4);
  font-size: 13.5px;
  padding: 48px 24px;
  text-align: center;
  border: 1px dashed var(--border2);
  border-radius: var(--radius);
  line-height: 1.6;
  font-family: 'Inter', system-ui, sans-serif;
}
.msg-idle strong { color: var(--text3); }

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
  .hero-title { font-size: 28px; }
}
"""


# ─────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  Result renderers
# ─────────────────────────────────────────────────────────────
def _summary_html(r: PipelineResult) -> str:
    if r.status == "error":
        errs = "<br>".join(r.errors) or "Unknown error."
        return f'<div class="msg-error"><b>Pipeline failed</b><br>{errs}</div>'

    changed = sum(1 for t in r.section_traces if t.changed)
    total   = len(r.section_traces)
    elapsed = f"{r.elapsed_ms / 1000:.1f}s"
    ats     = r.ats.score if r.ats else 0.0
    rev     = r.role_reviews[0].overall_score if r.role_reviews else 0.0
    jd_a    = r.jd_report.avg_score_after if r.jd_report else 0.0
    jd_d    = r.jd_report.avg_delta if r.jd_report else 0.0
    role    = ROLES.get(r.role, r.role)
    delta   = f"{jd_d:+.1f}" if jd_d else "—"

    kpis = "".join([
        _kpi("Sections", f"{changed}/{total}", "changed"),
        _kpi("ATS",      f"{ats:.0f}",          "/100"),
        _kpi("HM score", f"{rev:.0f}",           f"/100 · {role}"),
        _kpi("JD match", f"{jd_a:.0f}",          f"Δ {delta}"),
        _kpi("Time",     elapsed),
    ])

    warn = ""
    if r.warnings:
        warn = f'<div class="msg-warn">{"  ·  ".join(r.warnings[:3])}</div>'

    return f'<div class="kpi-grid">{kpis}</div>{warn}'


def _sections_html(r: PipelineResult) -> str:
    if not r.section_traces:
        return _idle("Section details appear here after enhancement.")
    out = []
    for t in r.section_traces:
        sp = _pill(f"{t.final_score:.0f}", "v" if t.final_score >= 75 else ("a" if t.final_score >= 50 else "r"))
        cp = _pill("changed", "g") if t.changed else _pill("unchanged", "z")

        dots = ""
        if t.iterations:
            d = "".join(
                f'<span class="iter-dot {"ok" if s.accepted else ("done" if s.verdict != "error" else "")}" '
                f'title="iter {s.iteration}: {s.score:.0f}"></span>'
                for s in t.iterations
            )
            dots = (
                f'<div class="iter-row">{d}'
                f'<span class="iter-meta">{t.iterations_used} iter · {t.final_score:.0f}/100</span>'
                f'</div>'
            )

        viols = ""
        if t.iterations and t.iterations[-1].violations:
            viols = f'<div class="violations">Critic: {"; ".join(t.iterations[-1].violations[:3])}</div>'

        note = f'<div class="trace-warn">{t.note}</div>' if t.note else ""

        out.append(
            f'<div class="trace-card">'
            f'<div class="trace-head">'
            f'<span class="trace-name">{t.label}</span>'
            f'<div class="trace-meta">{sp}{cp}</div>'
            f'</div>'
            f'<div class="diff-wrap">'
            f'<div class="diff-pane"><div class="diff-lbl">Before</div><div class="diff-txt">{t.before}</div></div>'
            f'<div class="diff-pane"><div class="diff-lbl">After</div><div class="diff-txt new">{t.after}</div></div>'
            f'</div>'
            f'{viols}{dots}{note}'
            f'</div>'
        )
    return "\n".join(out)


def _review_html(r: PipelineResult) -> str:
    if not r.role_reviews:
        return _idle("Hiring-manager review appears after enhancement.")
    rv = r.role_reviews[0]
    sc = _sc(rv.overall_score)

    strengths  = "".join(f"<li>{s}</li>" for s in rv.strengths)  or "<li><i>None</i></li>"
    weaknesses = "".join(f"<li>{w}</li>" for w in rv.weaknesses) or "<li><i>None</i></li>"
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
        f'<div class="review-col"><h5>Strengths</h5><ul>{strengths}</ul></div>'
        f'<div class="review-col weak"><h5>Weaknesses</h5><ul>{weaknesses}</ul></div>'
        f'</div>'
    )

    kw = f'<div class="kw-panel"><h5>Missing keywords</h5>{missing}</div>'

    extra = ""
    if len(r.role_reviews) > 1:
        rows = []
        for rr in r.role_reviews[1:]:
            kc = "v" if rr.overall_score >= 75 else ("a" if rr.overall_score >= 50 else "r")
            rows.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:9px 0;border-bottom:1px solid var(--border)">'
                f'<div><div style="font-size:13px;color:var(--text);font-weight:600">{rr.role_name}</div>'
                f'<div style="font-size:11.5px;color:var(--text3);margin-top:1px">"{rr.one_line_verdict}"</div></div>'
                f'{_pill(str(int(rr.overall_score)), kc)}'
                f'</div>'
            )
        extra = (
            f'<div style="background:var(--bg2);border:1px solid var(--border);'
            f'border-radius:var(--radius-sm);padding:14px 16px;margin-top:10px">'
            f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;'
            f'color:var(--text3);font-weight:700;margin-bottom:8px">Cross-role</div>'
            + "".join(rows) + "</div>"
        )

    return hero + cols + kw + extra


def _jd_html(r: PipelineResult) -> str:
    bits = []
    if r.jd_report and r.jd_report.samples:
        bits.append(_jd_block(r.jd_report, "Target role"))
    for jd in r.cross_role_jd_reports:
        bits.append(_jd_block(jd, "Cross-validation"))
    if not bits:
        return _idle("JD matching is off or no JDs loaded for this role.")
    return "\n".join(bits)


def _jd_block(rep, kind: str) -> str:
    role = ROLES.get(rep.role_id, rep.role_id)
    dk   = "v" if rep.avg_delta >= 5 else ("a" if rep.avg_delta >= 0 else "r")

    rows = [
        "<tr>"
        "<th>Job</th><th>Archetype</th>"
        "<th class='r'>Before</th><th class='r'>After</th>"
        "<th class='r'>Δ</th><th>Gaps</th>"
        "</tr>"
    ]
    for s in rep.samples:
        kl  = "g" if s.delta >= 5 else ("a" if s.delta >= 0 else "r")
        gaps = ", ".join(s.missing_keywords[:4]) or "—"
        rows.append(
            f'<tr>'
            f'<td><b style="color:var(--text);font-size:13px">{s.title}</b></td>'
            f'<td>{_pill(s.company_archetype, "z")}</td>'
            f'<td class="r">{s.score_before:.0f}</td>'
            f'<td class="r">{s.score_after:.0f}</td>'
            f'<td class="r">{_pill(f"{s.delta:+.1f}", kl)}</td>'
            f'<td class="dim">{gaps}</td>'
            f'</tr>'
        )

    hdr = (
        f'<div class="jd-header">'
        f'<span>{role}</span>'
        f'{_pill(kind, "z")}'
        f'<span style="margin-left:auto;font-size:13px;color:var(--text2);font-weight:400">'
        f'{rep.avg_score_before:.0f} → {rep.avg_score_after:.0f} '
        f'{_pill(f"{rep.avg_delta:+.1f}", dk)}'
        f'</span></div>'
    )

    gaps_html = ""
    if rep.top_gaps:
        gaps_html = (
            f'<div style="margin-top:12px">'
            f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;'
            f'color:var(--text3);font-weight:700;margin-bottom:7px">Top gaps</div>'
            + "".join(f'<span class="kw-tag miss">{k}</span>' for k in rep.top_gaps)
            + "</div>"
        )

    return (
        f'<div class="jd-section" style="background:var(--bg2);border:1px solid var(--border);'
        f'border-radius:var(--radius);padding:18px 20px">'
        f'{hdr}'
        f'<div class="jd-table-wrap"><table class="jd-table">{"".join(rows)}</table></div>'
        f'{gaps_html}</div>'
    )


# ─────────────────────────────────────────────────────────────
#  Progress
# ─────────────────────────────────────────────────────────────
_STAGES = [
    ("parse",    "📄", "Parse"),
    ("repair",   "🔧", "Repair"),
    ("plan",     "📐", "Plan"),
    ("enhance",  "✦",  "Enhance"),
    ("render",   "🖨", "Render"),
    ("jd_match", "🎯", "Validate"),
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

    log_html = "".join(f"<div>{ln}</div>" for ln in lines[-20:])

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
        name   = data.get("name", "")
        status = data.get("status", "")
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


# ─────────────────────────────────────────────────────────────
#  Handler  (only 2 real inputs: file + role)
# ─────────────────────────────────────────────────────────────
def _enhance_handler(file, role_id: str):
    blank = (
        _idle("Section details appear here."),
        _idle("Hiring-manager review appears here."),
        _idle("JD match scores appear here."),
        None, "",
        gr.update(visible=False),
    )

    if file is None:
        yield (_idle("Upload a <b>.tex</b> resume, pick a role, then hit <strong>Enhance</strong>."), *blank)
        return

    tex_path = Path(file.name if hasattr(file, "name") else file)
    if tex_path.suffix.lower() != ".tex":
        yield (_idle("Please upload a <b>.tex</b> file. PDF is not supported."), *blank)
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

    lines:        list[str]   = ['<span class="act">[start]</span> <span class="dim">initializing…</span>']
    pct:          int         = 3
    weights                   = {"parse": 10, "repair": 10, "complete": 8, "plan": 8,
                                 "enhance_plan": 4, "render": 15, "jd_match": 15}
    stages_done:  set[str]    = set()
    cur_stage:    str         = "parse"
    rew_done:     int         = 0
    rew_total:    int         = 0
    start_ts:     float       = time.monotonic()
    last_emit:    float       = time.monotonic()

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
        _sections_html(res),
        _review_html(res),
        _jd_html(res),
        res.tex_path,
        res.tex_content,
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
            'Upload a LaTeX resume. Our AI agents rewrite every bullet for impact — '
            'no hallucinations, Overleaf-ready output.'
            '</div>'
            '</div>'
        )

        # ── Input card ──
        gr.HTML(
            '<div class="main-card">'
            '<div class="card-header">'
            '<div class="card-header-icon">✦</div>'
            '<div class="card-header-title">Enhance Resume</div>'
            '<div class="card-header-sub">Upload · Select role · Run</div>'
            '</div>'
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                gr.HTML('<div style="padding:24px 24px 0">')

                file_in = gr.File(
                    label="Resume (.tex)",
                    file_types=[".tex"],
                    file_count="single",
                    height=120,
                )
                role_in = gr.Dropdown(
                    choices=role_choices,
                    value="ai_ml_engineer",
                    label="Target role",
                    interactive=True,
                )
                run_btn = gr.Button("✦  Enhance Resume", variant="primary", size="lg")

                gr.HTML('</div>')

            with gr.Column(scale=2):
                gr.HTML('<div style="padding:24px">')
                summary_out = gr.HTML(
                    _idle("Upload a <b>.tex</b> resume, select a role, then click <strong>✦ Enhance Resume</strong>.")
                )
                gr.HTML('</div>')

        gr.HTML('</div>')  # close .main-card

        # ── Results tabs ──
        with gr.Tabs():
            with gr.Tab("Sections"):
                sections_out = gr.HTML(_idle("Per-section before / after appears here."))

            with gr.Tab("Review"):
                review_out = gr.HTML(_idle("Hiring-manager review appears here."))

            with gr.Tab("JD Match"):
                jd_out = gr.HTML(_idle("JD keyword alignment scores appear here."))

            with gr.Tab("Download"):
                with gr.Group(visible=False) as dl_group:
                    gr.HTML(
                        '<div class="dl-hint">'
                        'Paste the .tex into <a href="https://overleaf.com" target="_blank">Overleaf</a> to compile to PDF.'
                        '</div>'
                    )
                    tex_file_out = gr.File(label="Enhanced .tex", interactive=False)
                    tex_text_out = gr.Code(label="Source preview", language="latex", lines=28, interactive=False)

        # ── Wire ──
        run_btn.click(
            _enhance_handler,
            inputs=[file_in, role_in],
            outputs=[summary_out, sections_out, review_out, jd_out, tex_file_out, tex_text_out, dl_group],
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
