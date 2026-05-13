"""
gradio_app.py — Neural Canvas Edition (v5)

Dark, animated, production-level UI for the Resume Enhancer multi-agent pipeline.

Tabs:
  - Enhance       upload .tex, configure, run pipeline with live stage animation.
  - Sections      per-section before/after with iteration trace.
  - Review        hiring-manager simulation for the target role.
  - JD Matching   keyword score against curated JDs.
  - Download      .tex output + Overleaf paste-ready helper.
  - Setup         backend status + auth instructions.

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

from app.core.config import settings                              # noqa: E402
from app.core.ir import PipelineResult                           # noqa: E402
from app.core.llm import (best_available_backend,               # noqa: E402
                          detect_available_backends,
                          is_backend_configured)
from app.core.skills import load_skills                          # noqa: E402
from app.pipeline import (PipelineConfig, ROLES,                # noqa: E402
                          list_role_keywords, run_pipeline)

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
        return (
            f"You've used all {max_runs} runs this hour. "
            f"Please wait ~{mins} min before trying again."
        )
    _rate_limit_store[session_id].append(now)
    return None


_ERROR_MAP = {
    "api key": "Your API key appears to be invalid or missing. Check the Setup tab.",
    "rate limit": "The AI service is temporarily busy. Please wait a moment and try again.",
    "timeout": "The AI service took too long to respond. Try again or switch backend.",
    "connection": "Could not connect to the AI service. Check your internet connection.",
    "not installed": "A required package is not installed. See the Setup tab.",
}


def _friendly_error(raw: str) -> str:
    lower = raw.lower()
    for key, msg in _ERROR_MAP.items():
        if key in lower:
            return msg
    return f"Something went wrong: {raw[:200]}. Please try again or switch backends."


# ─────────────────────────────────────────────────────────────
#  CSS — Neural Canvas Dark Theme
# ─────────────────────────────────────────────────────────────
CSS = """
/* ── Design tokens ─────────────────────────────────────────── */
:root {
  --nc-bg:          #07090f;
  --nc-bg-card:     rgba(14, 16, 30, 0.9);
  --nc-bg-card2:    rgba(20, 23, 42, 0.95);
  --nc-border:      rgba(124, 58, 237, 0.28);
  --nc-border-soft: rgba(255, 255, 255, 0.07);
  --nc-text:        #e2e8f0;
  --nc-text2:       #94a3b8;
  --nc-text3:       #475569;
  --nc-purple:      #7c3aed;
  --nc-purple-l:    #a78bfa;
  --nc-cyan:        #06b6d4;
  --nc-emerald:     #10b981;
  --nc-amber:       #f59e0b;
  --nc-rose:        #f43f5e;
  --nc-g-purple:    linear-gradient(135deg,#6d28d9,#7c3aed);
  --nc-g-cyan:      linear-gradient(135deg,#0369a1,#0ea5e9);
  --nc-g-emerald:   linear-gradient(135deg,#065f46,#059669);
  --nc-g-amber:     linear-gradient(135deg,#92400e,#d97706);
  --nc-g-rose:      linear-gradient(135deg,#9f1239,#e11d48);
  --nc-g-teal:      linear-gradient(135deg,#134e4a,#0f766e);
  --nc-g-slate:     linear-gradient(135deg,#1e293b,#334155);
  --nc-glow:        0 0 30px rgba(124,58,237,.22);
  --nc-glow-sm:     0 0 16px rgba(124,58,237,.28);
  --nc-r:           16px;
  --nc-r-sm:        12px;
  --nc-ease:        cubic-bezier(.2,.8,.2,1);
}

/* ── Base ───────────────────────────────────────────────────── */
* { box-sizing: border-box; }

.gradio-container {
  max-width: 1380px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Inter','Segoe UI',system-ui,sans-serif !important;
  font-feature-settings: 'cv11','ss01';
  letter-spacing: -.003em;
}

body, html { background: var(--nc-bg) !important; }

/* ── Animated background orbs ──────────────────────────────── */
.nc-orbs {
  position: fixed; inset: 0;
  pointer-events: none; z-index: -1; overflow: hidden;
}
.nc-orb {
  position: absolute; border-radius: 50%;
  filter: blur(90px); opacity: 0;
  animation: nc-reveal 2s ease forwards, nc-drift var(--dur,25s) ease-in-out var(--delay,0s) infinite;
}
.nc-orb-a {
  width: 720px; height: 720px;
  background: radial-gradient(circle,#7c3aed,transparent 70%);
  top: -280px; left: -180px;
  --dur: 28s; --delay: 0s;
}
.nc-orb-b {
  width: 600px; height: 600px;
  background: radial-gradient(circle,#06b6d4,transparent 70%);
  bottom: -220px; right: -160px;
  --dur: 22s; --delay: -8s;
}
.nc-orb-c {
  width: 500px; height: 500px;
  background: radial-gradient(circle,#10b981,transparent 70%);
  top: 45%; left: 42%;
  --dur: 32s; --delay: -16s;
}
@keyframes nc-reveal {
  from { opacity: 0; } to { opacity: .11; }
}
@keyframes nc-drift {
  0%,100% { transform: translate(0,0); }
  25%      { transform: translate(45px,-40px); }
  50%      { transform: translate(-30px,30px); }
  75%      { transform: translate(20px,50px); }
}

/* ── Hero ───────────────────────────────────────────────────── */
.nc-hero {
  position: relative; padding: 52px 48px 44px;
  margin: 10px 0 22px; border-radius: 24px; overflow: hidden;
  background: linear-gradient(135deg,#0b0d1e 0%,#131628 55%,#0e1124 100%);
  border: 1px solid rgba(124,58,237,.4);
  box-shadow: var(--nc-glow), inset 0 0 80px rgba(124,58,237,.06);
}
.nc-hero-grid {
  position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(124,58,237,.07) 1px,transparent 1px),
    linear-gradient(90deg,rgba(124,58,237,.07) 1px,transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(ellipse 90% 70% at 50% 0%,black 20%,transparent 75%);
}
.nc-hero-glow {
  position: absolute; inset: 0; pointer-events: none;
  background:
    radial-gradient(700px 280px at -8% -25%, rgba(124,58,237,.38),transparent 55%),
    radial-gradient(600px 260px at 110% -15%, rgba(6,182,212,.22),transparent 55%),
    radial-gradient(500px 180px at 55% 115%, rgba(16,185,129,.18),transparent 60%);
}
.nc-hero h1 {
  font-size: 44px; font-weight: 900; letter-spacing: -.045em;
  margin: 0 0 12px; line-height: 1;
  background: linear-gradient(95deg,#fff 0%,#c4b5fd 28%,#67e8f9 60%,#6ee7b7 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: nc-fade-up .8s var(--nc-ease) both;
}
.nc-hero .tagline {
  color: rgba(226,232,240,.72); font-size: 15px;
  line-height: 1.65; max-width: 700px; margin: 0 0 26px;
  animation: nc-fade-up .8s .1s var(--nc-ease) both;
}
.nc-hero .badges { display: flex; gap: 8px; flex-wrap: wrap; animation: nc-fade-up .8s .2s var(--nc-ease) both; }
@keyframes nc-fade-up {
  from { opacity:0; transform: translateY(14px); }
  to   { opacity:1; transform: translateY(0); }
}
.nc-badge {
  padding: 5px 14px; border-radius: 999px; font-size: 11px; font-weight: 600;
  letter-spacing: .04em; border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.05); backdrop-filter: blur(12px);
  color: rgba(255,255,255,.82);
  transition: all .22s var(--nc-ease); cursor: default;
}
.nc-badge:hover { background: rgba(255,255,255,.1); transform: translateY(-1px); }
.nc-badge.p { border-color: rgba(167,139,250,.4); color: #c4b5fd; background: rgba(124,58,237,.1); }
.nc-badge.c { border-color: rgba(103,232,249,.4); color: #67e8f9; background: rgba(6,182,212,.1); }
.nc-badge.e { border-color: rgba(110,231,183,.4); color: #6ee7b7; background: rgba(16,185,129,.1); }
.nc-hero-stats {
  position: absolute; top: 48px; right: 48px;
  display: flex; gap: 24px; align-items: flex-start;
}
.nc-hero-stat .val {
  font-size: 30px; font-weight: 900; letter-spacing: -.04em;
  background: linear-gradient(135deg,#c4b5fd,#a78bfa);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.nc-hero-stat .lbl { font-size: 9.5px; text-transform: uppercase; letter-spacing: .1em; color: rgba(255,255,255,.35); margin-top: 1px; }

/* ── Glass card ─────────────────────────────────────────────── */
.nc-card {
  background: var(--nc-bg-card);
  backdrop-filter: blur(24px);
  border: 1px solid var(--nc-border-soft);
  border-radius: var(--nc-r); padding: 20px 22px;
  transition: border-color .22s, box-shadow .22s;
}
.nc-card:hover { border-color: var(--nc-border); }
.nc-card-title {
  font-size: 13px; font-weight: 700; color: var(--nc-text);
  letter-spacing: -.01em; margin: 0 0 5px;
  display: flex; align-items: center; gap: 8px;
}
.nc-step {
  display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px; border-radius: 8px;
  background: var(--nc-g-purple); color: #fff;
  font-size: 11px; font-weight: 800; flex-shrink: 0;
  box-shadow: var(--nc-glow-sm);
}
.nc-card-sub { font-size: 12.5px; color: var(--nc-text2); line-height: 1.5; margin: 0; }

/* ── KPI grid ───────────────────────────────────────────────── */
.nc-kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit,minmax(155px,1fr));
  gap: 12px; margin-bottom: 20px;
}
.nc-kpi {
  padding: 18px 18px 14px; border-radius: 16px; overflow: hidden;
  position: relative; isolation: isolate; cursor: default;
  transition: transform .25s var(--nc-ease), box-shadow .25s var(--nc-ease);
}
.nc-kpi:hover { transform: translateY(-3px); }
.nc-kpi::after {
  content:''; position:absolute; top:-35px; right:-35px;
  width:110px; height:110px; border-radius:50%;
  background: rgba(255,255,255,.07); pointer-events:none;
}
.nc-kpi .kv {
  font-size: 32px; font-weight: 900; letter-spacing: -.04em;
  line-height: 1; margin-bottom: 4px; color: #fff;
  font-variant-numeric: tabular-nums;
}
.nc-kpi .kl {
  font-size: 9.5px; text-transform: uppercase; letter-spacing: .11em;
  color: rgba(255,255,255,.72); font-weight: 700;
}
.nc-kpi .kd { font-size: 10.5px; color: rgba(255,255,255,.58); margin-top: 6px; }
.nc-kpi.purple  { background: linear-gradient(145deg,#5b21b6,#7c3aed); box-shadow: 0 8px 28px -8px rgba(124,58,237,.55); }
.nc-kpi.cyan    { background: linear-gradient(145deg,#0369a1,#0ea5e9); box-shadow: 0 8px 28px -8px rgba(14,165,233,.5); }
.nc-kpi.emerald { background: linear-gradient(145deg,#065f46,#059669); box-shadow: 0 8px 28px -8px rgba(16,185,129,.5); }
.nc-kpi.amber   { background: linear-gradient(145deg,#92400e,#d97706); box-shadow: 0 8px 28px -8px rgba(245,158,11,.5); }
.nc-kpi.rose    { background: linear-gradient(145deg,#9f1239,#e11d48); box-shadow: 0 8px 28px -8px rgba(244,63,94,.5); }
.nc-kpi.slate   { background: linear-gradient(145deg,#1e293b,#334155); box-shadow: 0 8px 28px -8px rgba(51,65,85,.45); }
.nc-kpi.teal    { background: linear-gradient(145deg,#134e4a,#0f766e); box-shadow: 0 8px 28px -8px rgba(20,184,166,.5); }

/* ── Score bar ──────────────────────────────────────────────── */
.nc-bar { height: 5px; background: rgba(255,255,255,.08); border-radius:999px; overflow:hidden; }
.nc-bar i { display:block; height:100%; border-radius:999px; transition: width .65s var(--nc-ease); }
.nc-bar.high i { background: linear-gradient(90deg,#10b981,#34d399); box-shadow: 0 0 10px #10b981; }
.nc-bar.mid  i { background: linear-gradient(90deg,#f59e0b,#fbbf24); box-shadow: 0 0 10px #f59e0b; }
.nc-bar.low  i { background: linear-gradient(90deg,#f43f5e,#fb7185); box-shadow: 0 0 10px #f43f5e; }
.nc-bar.def  i { background: linear-gradient(90deg,#7c3aed,#a855f7); box-shadow: 0 0 10px #7c3aed; }

/* ── Score badges ───────────────────────────────────────────── */
.nc-s {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
  letter-spacing: .03em; font-variant-numeric: tabular-nums;
}
.nc-s.high    { background: rgba(16,185,129,.18); color: #6ee7b7; border: 1px solid rgba(16,185,129,.35); }
.nc-s.mid     { background: rgba(245,158,11,.18); color: #fde68a; border: 1px solid rgba(245,158,11,.35); }
.nc-s.low     { background: rgba(244,63,94,.18);  color: #fca5a5; border: 1px solid rgba(244,63,94,.35); }
.nc-s.changed { background: rgba(124,58,237,.2);  color: #c4b5fd; border: 1px solid rgba(124,58,237,.35); }
.nc-s.unch    { background: rgba(255,255,255,.06); color: var(--nc-text3); border: 1px solid rgba(255,255,255,.1); }
.nc-s.iter    { background: rgba(167,139,250,.15); color: #c4b5fd; border: 1px solid rgba(167,139,250,.3); }

/* ── Tags ───────────────────────────────────────────────────── */
.nc-tag {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 600; margin: 2px 3px 2px 0;
  background: rgba(124,58,237,.14); color: #c4b5fd;
  border: 1px solid rgba(124,58,237,.3);
  transition: transform .14s;
}
.nc-tag:hover { transform: translateY(-1px); }
.nc-tag.miss  { background: rgba(244,63,94,.14);  color: #fca5a5; border-color: rgba(244,63,94,.3); }
.nc-tag.match { background: rgba(16,185,129,.14); color: #6ee7b7; border-color: rgba(16,185,129,.3); }

/* ── Pipeline progress ──────────────────────────────────────── */
.nc-pipeline {
  padding: 22px 24px; border-radius: var(--nc-r);
  background: var(--nc-bg-card2);
  border: 1px solid var(--nc-border-soft);
}
.nc-pipeline-hd {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
}
.nc-pipeline-title { font-size: 14px; font-weight: 700; color: var(--nc-text); }
.nc-pipeline-eta   { font-size: 12px; color: var(--nc-text2); }
.nc-prog-bar { height: 4px; background: rgba(255,255,255,.07); border-radius:999px; overflow:hidden; margin-bottom:20px; }
.nc-prog-fill {
  height: 100%; border-radius: 999px; transition: width .55s var(--nc-ease);
  background: linear-gradient(90deg,#6d28d9,#7c3aed,#a855f7,#06b6d4);
  background-size: 200% 100%; animation: nc-shimmer 2.5s linear infinite;
  box-shadow: 0 0 14px rgba(124,58,237,.6);
}
@keyframes nc-shimmer {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Stage nodes */
.nc-stages { display: flex; align-items: center; margin-bottom: 18px; gap: 0; }
.nc-stage-wrap { display: flex; flex-direction: column; align-items: center; gap: 5px; flex: 1; }
.nc-stage-dot {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
  border: 2px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.04);
  transition: all .35s var(--nc-ease);
  position: relative; z-index: 1;
}
.nc-stage-dot.idle   { opacity: .45; }
.nc-stage-dot.active {
  background: rgba(124,58,237,.28); border-color: #7c3aed;
  box-shadow: 0 0 18px rgba(124,58,237,.55);
  animation: nc-pulse 1.6s ease-in-out infinite;
}
.nc-stage-dot.done   {
  background: rgba(16,185,129,.2); border-color: #10b981;
  box-shadow: 0 0 14px rgba(16,185,129,.4);
}
@keyframes nc-pulse {
  0%,100% { box-shadow: 0 0 18px rgba(124,58,237,.55); }
  50%      { box-shadow: 0 0 32px rgba(124,58,237,.85), 0 0 0 7px rgba(124,58,237,.12); }
}
.nc-stage-lbl {
  font-size: 9px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--nc-text3); font-weight: 600; text-align: center;
  transition: color .3s;
}
.nc-stage-lbl.active { color: #c4b5fd; }
.nc-stage-lbl.done   { color: #6ee7b7; }
.nc-stage-conn {
  flex: 0 0 auto; width: 28px; height: 1px; margin-bottom: 19px;
  background: rgba(255,255,255,.1); transition: background .4s;
}
.nc-stage-conn.done { background: rgba(16,185,129,.5); }

/* Log area */
.nc-log {
  background: rgba(0,0,0,.45); border-radius: 10px; padding: 12px 14px;
  font-family: 'JetBrains Mono',ui-monospace,Consolas,monospace;
  font-size: 11.5px; line-height: 1.75; color: #94a3b8;
  max-height: 320px; overflow-y: auto;
  border: 1px solid rgba(255,255,255,.05);
}
.nc-log .ls  { color: #a78bfa; font-weight: 700; }
.nc-log .lok { color: #6ee7b7; }
.nc-log .lw  { color: #fde68a; }
.nc-log .le  { color: #fca5a5; }
.nc-log .ld  { color: #475569; }
.nc-log::-webkit-scrollbar { width: 5px; }
.nc-log::-webkit-scrollbar-thumb { background: rgba(124,58,237,.35); border-radius: 3px; }

/* ── Section traces ─────────────────────────────────────────── */
.nc-trace { margin-bottom: 14px; }
.nc-trace-card {
  border: 1px solid var(--nc-border-soft); border-radius: var(--nc-r);
  overflow: hidden; background: var(--nc-bg-card);
  transition: border-color .2s;
}
.nc-trace-card:hover { border-color: var(--nc-border); }
.nc-trace-hd {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,.05);
  background: rgba(255,255,255,.02);
}
.nc-trace-lbl { font-weight: 700; font-size: 13px; color: var(--nc-text); }
.nc-trace-meta { display: flex; gap: 6px; align-items: center; }
.nc-diff-before {
  padding: 13px 16px; font-size: 13px; line-height: 1.7;
  background: rgba(245,158,11,.06);
  border-left: 3px solid rgba(245,158,11,.5);
  color: rgba(253,230,138,.9);
  border-bottom: 1px solid rgba(255,255,255,.04);
}
.nc-diff-after {
  padding: 13px 16px; font-size: 13px; line-height: 1.7;
  background: rgba(16,185,129,.06);
  border-left: 3px solid rgba(16,185,129,.5);
  color: rgba(110,231,183,.9);
}
.nc-diff-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; opacity: .45; margin-right: 8px; }
.nc-flags {
  margin: 8px 16px; padding: 9px 12px; border-radius: 9px;
  background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.25);
  color: #fde68a; font-size: 12px; line-height: 1.5;
}
.nc-iter-row { display: flex; gap: 5px; padding: 8px 16px; align-items: center; }
.nc-iter-dot { width: 7px; height: 7px; border-radius: 50%; background: rgba(255,255,255,.14); transition: all .3s; }
.nc-iter-dot.done   { background: #6ee7b7; }
.nc-iter-dot.accept { background: #a78bfa; box-shadow: 0 0 8px rgba(167,139,250,.7); }
.nc-iter-dim { font-size: 11px; color: var(--nc-text3); margin-left: 6px; }
.nc-trace-note { font-size: 11.5px; color: var(--nc-text3); font-style: italic; padding: 4px 16px 10px; }

/* ── Review ─────────────────────────────────────────────────── */
.nc-review-hero {
  border-radius: var(--nc-r); padding: 28px 30px 22px;
  background: linear-gradient(135deg,rgba(11,13,28,.98),rgba(18,21,40,.98));
  border: 1px solid rgba(124,58,237,.35); margin-bottom: 14px;
  position: relative; overflow: hidden;
  box-shadow: var(--nc-glow);
}
.nc-review-hero::before {
  content:''; position:absolute; top:-60px; right:-60px; width:280px; height:280px;
  border-radius:50%;
  background: radial-gradient(circle,rgba(124,58,237,.2),transparent 65%);
  pointer-events:none;
}
.nc-review-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.nc-review-h2  { font-size: 22px; font-weight: 800; letter-spacing: -.025em; color: var(--nc-text); margin: 0 0 5px; }
.nc-review-sub { color: var(--nc-text2); font-size: 13px; }
.nc-score-box  { text-align: right; }
.nc-score-num  {
  font-size: 54px; font-weight: 900; letter-spacing: -.05em; line-height: 1;
  font-variant-numeric: tabular-nums;
}
.nc-score-num.high { background: var(--nc-g-emerald); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.nc-score-num.mid  { background: var(--nc-g-amber);   -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.nc-score-num.low  { background: var(--nc-g-rose);    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.nc-score-lbl { font-size: 9.5px; text-transform: uppercase; letter-spacing: .1em; color: var(--nc-text3); font-weight: 700; margin-top: 4px; }
.nc-verdict {
  margin-top: 16px; padding: 14px 16px; border-radius: 12px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.07);
  border-left: 3px solid #7c3aed;
  font-style: italic; color: var(--nc-text2); font-size: 14px; line-height: 1.55;
}
.nc-review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0; }
.nc-review-col {
  background: var(--nc-bg-card); border: 1px solid var(--nc-border-soft);
  border-radius: var(--nc-r-sm); padding: 16px 18px;
}
.nc-review-col h5 {
  font-size: 10px; text-transform: uppercase; letter-spacing: .12em;
  margin: 0 0 12px; color: var(--nc-text3); font-weight: 800;
  display: flex; align-items: center; gap: 6px;
}
.nc-review-col h5::before { content:''; width:4px; height:14px; border-radius:2px; background:#6ee7b7; flex-shrink:0; }
.nc-review-col.weak h5::before { background: #fde68a; }
.nc-review-col ul { margin:0; padding:0; list-style:none; }
.nc-review-col ul li {
  font-size: 13px; color: var(--nc-text); line-height: 1.55;
  margin-bottom: 6px; padding-left: 16px; position: relative;
}
.nc-review-col ul li::before { content:'▸'; position:absolute; left:0; color:#6ee7b7; font-size:11px; top:1px; }
.nc-review-col.weak ul li::before { content:'◦'; color:#fde68a; }
.nc-kw-card {
  background: var(--nc-bg-card); border: 1px solid var(--nc-border-soft);
  border-radius: var(--nc-r-sm); padding: 14px 16px;
}
.nc-kw-card h5 {
  font-size: 10px; text-transform: uppercase; letter-spacing: .12em;
  margin: 0 0 10px; color: var(--nc-text3); font-weight: 800;
}

/* ── JD table ───────────────────────────────────────────────── */
.nc-table { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
.nc-table-wrap {
  border-radius: var(--nc-r-sm); overflow: hidden;
  border: 1px solid rgba(255,255,255,.06);
  background: var(--nc-bg-card);
}
.nc-table th, .nc-table td {
  padding: 12px 14px; text-align: left;
  border-bottom: 1px solid rgba(255,255,255,.05);
  color: var(--nc-text);
}
.nc-table th {
  background: rgba(255,255,255,.03); color: var(--nc-text3);
  font-size: 9.5px; text-transform: uppercase; letter-spacing: .1em; font-weight: 800;
}
.nc-table tr:last-child td { border-bottom: none; }
.nc-table tr:hover td { background: rgba(124,58,237,.06); }
.nc-table td.num { font-variant-numeric: tabular-nums; font-weight: 600; }
.nc-table td.gap { font-size: 12px; color: var(--nc-text2); }

/* ── Setup ──────────────────────────────────────────────────── */
.nc-setup-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(290px,1fr)); gap: 14px; }
.nc-setup-card {
  background: var(--nc-bg-card); border: 1px solid var(--nc-border-soft);
  border-radius: var(--nc-r); padding: 22px 22px 18px;
  transition: all .22s var(--nc-ease); position: relative; overflow: hidden;
}
.nc-setup-card:hover { border-color: var(--nc-border); transform: translateY(-2px); box-shadow: var(--nc-glow-sm); }
.nc-setup-card.ready {
  border-color: rgba(16,185,129,.38);
  background: linear-gradient(135deg,rgba(16,185,129,.09),rgba(5,150,105,.04));
}
.nc-setup-card.ready::after {
  content: '✓ ACTIVE'; position: absolute; top: 14px; right: 14px;
  font-size: 9px; font-weight: 800; letter-spacing: .12em; color: #6ee7b7;
  background: rgba(16,185,129,.18); padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(16,185,129,.35);
}
.nc-setup-card h4 { margin: 0 0 6px; font-size: 16px; color: var(--nc-text); font-weight: 700; }
.nc-setup-card .why { color: var(--nc-text2); font-size: 12.5px; margin: 0 0 12px; line-height: 1.5; }
.nc-setup-card .steps {
  background: rgba(0,0,0,.55); color: #94a3b8; padding: 12px 14px;
  border-radius: 10px; font-family: 'JetBrains Mono',ui-monospace,monospace;
  font-size: 11.5px; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
  border: 1px solid rgba(255,255,255,.05);
}
.nc-setup-card .cost {
  display: inline-block; margin-top: 10px;
  font-size: 9.5px; color: var(--nc-text3); font-weight: 700;
  text-transform: uppercase; letter-spacing: .1em;
}

/* ── Bento tiles ────────────────────────────────────────────── */
.nc-bento { display: grid; grid-template-columns: repeat(12,1fr); gap: 12px; margin-bottom: 18px; }
.nc-tile {
  border-radius: 14px; padding: 16px 18px; color: #fff;
  min-height: 96px; position: relative; overflow: hidden;
  transition: transform .25s var(--nc-ease);
}
.nc-tile:hover { transform: translateY(-2px); }
.nc-tile::before {
  content:''; position:absolute; top:-30px; right:-30px;
  width:110px; height:110px; border-radius:50%;
  background:rgba(255,255,255,.07); pointer-events:none;
}
.nc-tile h4 { margin: 0 0 7px; font-size: 13.5px; font-weight: 700; }
.nc-tile p  { margin: 0; font-size: 12px; line-height: 1.45; opacity: .85; }
.nt1 { grid-column:span 5; background:linear-gradient(135deg,#5b21b6,#7c3aed); box-shadow:0 8px 24px -8px rgba(124,58,237,.5); }
.nt2 { grid-column:span 4; background:linear-gradient(135deg,#0369a1,#0ea5e9); box-shadow:0 8px 24px -8px rgba(14,165,233,.5); }
.nt3 { grid-column:span 3; background:linear-gradient(135deg,#9f1239,#e11d48); box-shadow:0 8px 24px -8px rgba(225,29,72,.5); }
.nt4 { grid-column:span 7; background:linear-gradient(135deg,#134e4a,#0f766e); box-shadow:0 8px 24px -8px rgba(20,184,166,.5); }
.nt5 { grid-column:span 5; background:linear-gradient(135deg,#7c2d12,#ea580c); box-shadow:0 8px 24px -8px rgba(234,88,12,.5); }

/* ── CTA button ─────────────────────────────────────────────── */
button.nc-cta {
  width:100% !important; padding:16px 28px !important;
  font-size:15px !important; font-weight:800 !important;
  letter-spacing:-.01em !important; border-radius:14px !important;
  background:linear-gradient(135deg,#5b21b6,#7c3aed) !important;
  border:none !important; color:#fff !important;
  box-shadow:0 8px 28px -8px rgba(109,40,217,.65) !important;
  transition:all .25s var(--nc-ease) !important;
  position:relative !important; overflow:hidden !important;
}
button.nc-cta::before {
  content:''; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(135deg,rgba(255,255,255,.12),transparent 50%);
}
button.nc-cta:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 14px 36px -8px rgba(109,40,217,.75) !important;
  background:linear-gradient(135deg,#6d28d9,#a855f7) !important;
}
button.nc-cta:active { transform:translateY(0) !important; }

/* ── Tabs dark ──────────────────────────────────────────────── */
.gradio-container .tab-nav {
  background: rgba(14,16,30,.92) !important;
  border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 14px !important; padding: 6px !important;
  backdrop-filter: blur(20px) !important;
  margin-bottom: 18px !important;
}
.gradio-container .tab-nav button {
  color: var(--nc-text2) !important; border-radius: 10px !important;
  font-weight: 600 !important; font-size: 13px !important;
  transition: all .2s !important; letter-spacing: -.005em !important;
}
.gradio-container .tab-nav button.selected {
  background: linear-gradient(135deg,#5b21b6,#7c3aed) !important;
  color: #fff !important;
  box-shadow: 0 4px 16px -4px rgba(109,40,217,.55) !important;
}
.gradio-container .tab-nav button:hover:not(.selected) {
  background: rgba(255,255,255,.06) !important;
  color: var(--nc-text) !important;
}

/* ── Gradio input overrides (dark) ──────────────────────────── */
.gradio-container input,
.gradio-container textarea {
  background: rgba(255,255,255,.05) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  color: var(--nc-text) !important;
  border-radius: 12px !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: rgba(124,58,237,.55) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,.15) !important;
  outline: none !important;
}
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .prose { color: var(--nc-text2) !important; }
.gradio-container .block { background: transparent !important; }
.gradio-container .panel { background: transparent !important; }
.gradio-container .wrap { background: transparent !important; }
.gradio-container .form { background: transparent !important; }

/* Dropdown / select */
.gradio-container select,
.gradio-container .svelte-select { background: rgba(255,255,255,.05) !important; color: var(--nc-text) !important; border-color: rgba(255,255,255,.1) !important; border-radius: 12px !important; }
.gradio-container .svelte-select .item { color: var(--nc-text) !important; }
.gradio-container .svelte-select .listContainer { background: #0d0f1e !important; border-color: rgba(255,255,255,.1) !important; }

/* Radio */
.gradio-container .radio-group { background: rgba(255,255,255,.03) !important; border-radius: 12px !important; padding: 8px !important; }
.gradio-container .wrap.svelte-1sbfox4 { background: transparent !important; }

/* File upload */
.gradio-container .upload-button,
.gradio-container .file-preview {
  background: rgba(124,58,237,.1) !important;
  border: 2px dashed rgba(124,58,237,.4) !important;
  border-radius: 14px !important; color: #c4b5fd !important;
}
.gradio-container .upload-button:hover { background: rgba(124,58,237,.18) !important; border-color: rgba(124,58,237,.65) !important; }

/* Accordion */
.gradio-container .accordion {
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 12px !important;
}
.gradio-container details > summary { color: var(--nc-text2) !important; }

/* Code blocks */
.gradio-container .code-wrap { background: rgba(0,0,0,.4) !important; border-color: rgba(255,255,255,.07) !important; }

/* ── Empty state ────────────────────────────────────────────── */
.nc-empty {
  text-align:center; color:var(--nc-text2); padding:44px 24px;
  background:rgba(255,255,255,.03);
  border:1px dashed rgba(255,255,255,.1); border-radius:var(--nc-r);
  font-size:13.5px; line-height:1.5;
}
.nc-empty .big { font-size:42px; margin-bottom:10px; opacity:.3; display:block; }

/* ── Error/warning ──────────────────────────────────────────── */
.nc-err {
  background: rgba(244,63,94,.1); border: 1px solid rgba(244,63,94,.3);
  border-radius: var(--nc-r); padding: 18px 20px;
}
.nc-err .et { font-weight:700; color:#fca5a5; margin-bottom:5px; font-size:14px; }
.nc-err .eb { color:#fca5a5; opacity:.8; font-size:13px; line-height:1.5; }
.nc-warn {
  background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.28);
  border-radius: var(--nc-r); padding: 14px 16px; margin-top: 12px;
}
.nc-warn .wt { font-weight:700; color:#fde68a; margin-bottom:4px; font-size:13px; }
.nc-warn ul { margin: 4px 0 0 18px; color:#fde68a; opacity:.8; font-size:12px; }

/* ── Footer ─────────────────────────────────────────────────── */
.nc-footer {
  text-align:center; color:var(--nc-text3); font-size:12px;
  margin:22px 0 10px; padding-top:16px;
  border-top: 1px solid rgba(255,255,255,.05);
}

/* ── Scrollbar global ───────────────────────────────────────── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.03); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,.32); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,.55); }

/* ── Responsive ─────────────────────────────────────────────── */
@media (max-width: 900px) {
  .nc-hero h1 { font-size: 30px; }
  .nc-hero-stats { display: none; }
  .nc-bento { grid-template-columns: 1fr 1fr; }
  .nt1,.nt2,.nt3,.nt4,.nt5 { grid-column: span 2; }
  .nc-review-grid { grid-template-columns: 1fr; }
  .nc-kpi-grid { grid-template-columns: repeat(3,1fr); }
}
"""


# ─────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────
def _sc(score: float) -> str:
    if score >= 80: return "high"
    if score >= 60: return "mid"
    return "low"


def _kpi(label: str, value: str, klass: str = "", delta: str = "") -> str:
    delta_html = f'<div class="kd">{delta}</div>' if delta else ""
    return (
        f'<div class="nc-kpi {klass}">'
        f'<div class="kv">{value}</div>'
        f'<div class="kl">{label}</div>'
        f'{delta_html}'
        '</div>'
    )


def _empty(icon: str, msg: str) -> str:
    return f'<div class="nc-empty"><span class="big">{icon}</span>{msg}</div>'


def _bar(score: float, max_score: float = 100.0) -> str:
    pct = max(0.0, min(100.0, (score / max_score) * 100.0)) if max_score else 0.0
    cls = _sc(score if max_score == 100 else (score / max_score) * 100.0)
    return f'<div class="nc-bar {cls}"><i style="width:{pct:.0f}%"></i></div>'


# ─────────────────────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────────────────────
def _hero_html() -> str:
    return """
<div class="nc-orbs">
  <div class="nc-orb nc-orb-a"></div>
  <div class="nc-orb nc-orb-b"></div>
  <div class="nc-orb nc-orb-c"></div>
</div>
<div class="nc-hero">
  <div class="nc-hero-grid"></div>
  <div class="nc-hero-glow"></div>
  <h1>AI Resume Enhancer</h1>
  <p class="tagline">
    Upload your <b>.tex</b> resume, choose a target role, and a multi-agent pipeline will rewrite every
    section with role-tuned keywords, ATS alignment, and hiring-manager quality checks —
    without inventing or dropping any of your facts.
  </p>
  <div class="badges">
    <span class="nc-badge p">Multi-Agent</span>
    <span class="nc-badge c">Fact-Preserving</span>
    <span class="nc-badge e">ATS-Optimized</span>
    <span class="nc-badge p">Overleaf-Ready</span>
    <span class="nc-badge c">Role-Aligned</span>
    <span class="nc-badge e">Critic-Reviewed</span>
  </div>
  <div class="nc-hero-stats">
    <div class="nc-hero-stat"><div class="val">8</div><div class="lbl">Agents</div></div>
    <div class="nc-hero-stat"><div class="val">5</div><div class="lbl">Safety Guards</div></div>
    <div class="nc-hero-stat"><div class="val">∞</div><div class="lbl">Roles</div></div>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────
#  Bento feature tiles
# ─────────────────────────────────────────────────────────────
def _bento_html() -> str:
    return """
<div class="nc-bento">
  <div class="nc-tile nt1">
    <h4>Role-Tuned Engine</h4>
    <p>Restructures and rewrites every section for your chosen target role, adapting order, tone and keyword emphasis.</p>
  </div>
  <div class="nc-tile nt2">
    <h4>Zero Data Loss</h4>
    <p>Protected terms, numbers and facts are extracted before any rewrite and verified after.</p>
  </div>
  <div class="nc-tile nt3">
    <h4>Live Pipeline</h4>
    <p>Watch each agent stage complete in real-time with animated progress and estimated wait.</p>
  </div>
  <div class="nc-tile nt4">
    <h4>Market Validation — JD Match + Hiring-Manager Sim</h4>
    <p>Your enhanced resume is scored against curated job descriptions and reviewed through a simulated hiring-manager lens for each target role.</p>
  </div>
  <div class="nc-tile nt5">
    <h4>Overleaf Output</h4>
    <p>Download the .tex and compile to PDF directly in Overleaf — no local LaTeX needed.</p>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────
#  Result panels
# ─────────────────────────────────────────────────────────────
def _summary_html(result: PipelineResult) -> str:
    if result.status == "error":
        errs = "<br>".join(result.errors) or "Unknown error."
        return (
            '<div class="nc-err">'
            '<div class="et">Pipeline failed</div>'
            f'<div class="eb">{errs}</div>'
            '</div>'
        )

    sections_changed = sum(1 for t in result.section_traces if t.changed)
    sections_total   = len(result.section_traces)
    elapsed          = f"{result.elapsed_ms / 1000:.1f}s"
    ats              = result.ats.score if result.ats else 0.0
    review_score     = result.role_reviews[0].overall_score if result.role_reviews else 0.0
    jd_delta         = result.jd_report.avg_delta if result.jd_report else 0.0
    jd_after         = result.jd_report.avg_score_after if result.jd_report else 0.0
    delta_str        = f"{jd_delta:+.1f} vs original" if jd_delta else ""
    role_pretty      = ROLES.get(result.role, result.role)
    backend_pretty   = {"claude_code": "Claude Code", "anthropic": "Anthropic API", "huggingface": "HuggingFace"}.get(result.backend, result.backend)

    kpis = "".join([
        _kpi("Status",          result.status.upper(),             "emerald" if result.status == "complete" else "amber"),
        _kpi("Sections",        f"{sections_changed}/{sections_total}", "purple"),
        _kpi("ATS Score",       f"{ats:.0f}/100",                   _sc(ats)),
        _kpi("Hiring Manager",  f"{review_score:.0f}/100",          _sc(review_score), f"target: {role_pretty}"),
        _kpi("JD Match",        f"{jd_after:.0f}/100",              "teal",            delta_str),
        _kpi("Elapsed",         elapsed,                            "slate",           f"via {backend_pretty}"),
    ])

    warnings = ""
    if result.warnings:
        ws = "".join(f"<li>{w}</li>" for w in result.warnings)
        warnings = (
            '<div class="nc-warn" style="margin-top:14px">'
            '<div class="wt">Notes</div>'
            f'<ul>{ws}</ul></div>'
        )

    return f'<div class="nc-kpi-grid">{kpis}</div>{warnings}'


def _sections_html(result: PipelineResult) -> str:
    if not result.section_traces:
        return _empty("⬡", "Per-section before/after traces will appear here once the pipeline runs.")
    rows = []
    for t in result.section_traces:
        sc   = _sc(t.final_score)
        flags = ""
        if t.iterations:
            last = t.iterations[-1]
            if last.violations:
                flags = (
                    '<div class="nc-flags"><b>Critic flags:</b> '
                    + "; ".join(last.violations[:4]) + '</div>'
                )
        iter_dots = []
        for s in t.iterations:
            cls = "accept" if s.accepted else ("done" if s.verdict != "error" else "")
            iter_dots.append(f'<span class="nc-iter-dot {cls}" title="iter {s.iteration}: {s.score:.0f}"></span>')
        iter_row = (
            '<div class="nc-iter-row">'
            + "".join(iter_dots)
            + f'<span class="nc-iter-dim">{t.iterations_used} iter · final {t.final_score:.0f}/100</span>'
            + '</div>'
        ) if iter_dots else ""
        note = (
            f'<div class="nc-trace-note">{t.note}</div>'
        ) if t.note else ""
        change_badge = (
            '<span class="nc-s changed">CHANGED</span>'
            if t.changed else
            '<span class="nc-s unch">UNCHANGED</span>'
        )
        rows.append(
            '<div class="nc-trace-card">'
            '<div class="nc-trace-hd">'
            f'<span class="nc-trace-lbl">{t.label}</span>'
            f'<div class="nc-trace-meta">'
            f'<span class="nc-s {sc}">{t.final_score:.0f}</span>'
            f'{change_badge}'
            '</div></div>'
            f'<div class="nc-diff-before"><span class="nc-diff-lbl">Before</span>{t.before}</div>'
            f'<div class="nc-diff-after"><span class="nc-diff-lbl">After</span>{t.after}</div>'
            f'{flags}{iter_row}{note}'
            '</div>'
        )
    return "\n".join(rows)


def _review_html(result: PipelineResult) -> str:
    if not result.role_reviews:
        return _empty("◑", "Hiring-manager review will appear here after enhancement.")
    r  = result.role_reviews[0]
    sc = _sc(r.overall_score)
    strengths  = "".join(f"<li>{s}</li>" for s in r.strengths)  or "<li><i>none returned</i></li>"
    weaknesses = "".join(f"<li>{w}</li>" for w in r.weaknesses) or "<li><i>none</i></li>"
    missing    = "".join(f'<span class="nc-tag miss">{k}</span>' for k in r.missing_keywords) or '<span class="nc-tag">none</span>'

    hero = (
        '<div class="nc-review-hero">'
        '<div class="nc-review-top">'
        '<div>'
        f'<h2 class="nc-review-h2">{r.role_name}</h2>'
        '<p class="nc-review-sub">Simulated hiring-manager read against this role profile</p>'
        '</div>'
        '<div class="nc-score-box">'
        f'<div class="nc-score-num {sc}">{r.overall_score:.0f}</div>'
        '<div class="nc-score-lbl">Phone-screen<br>likelihood</div>'
        '</div>'
        '</div>'
        f'<div style="margin-top:16px">{_bar(r.overall_score)}</div>'
        f'<div class="nc-verdict">"{r.one_line_verdict}"</div>'
        '</div>'
    )
    grid = (
        '<div class="nc-review-grid">'
        f'<div class="nc-review-col"><h5>Strengths</h5><ul>{strengths}</ul></div>'
        f'<div class="nc-review-col weak"><h5>Weaknesses</h5><ul>{weaknesses}</ul></div>'
        '</div>'
    )
    kw = (
        '<div class="nc-kw-card">'
        '<h5>Missing high-impact keywords</h5>'
        f'{missing}'
        '</div>'
    )

    extra = ""
    if len(result.role_reviews) > 1:
        chips = []
        for rr in result.role_reviews[1:]:
            kc = _sc(rr.overall_score)
            chips.append(
                '<div class="nc-card" style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;margin-top:10px">'
                f'<div><b style="color:var(--nc-text)">{rr.role_name}</b>'
                f'<div style="font-size:12px;color:var(--nc-text3);margin-top:2px">"{rr.one_line_verdict}"</div></div>'
                f'<span class="nc-s {kc}">{rr.overall_score:.0f}</span></div>'
            )
        extra = (
            '<div style="margin-top:16px">'
            '<div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--nc-text3);font-weight:800;margin-bottom:4px">Cross-role</div>'
            + "".join(chips) + "</div>"
        )
    return hero + grid + kw + extra


def _jd_html(result: PipelineResult) -> str:
    bits = []
    if result.jd_report and result.jd_report.samples:
        bits.append(_jd_table(result.jd_report, "Target role"))
    for r in result.cross_role_jd_reports:
        bits.append(_jd_table(r, "Cross-validation"))
    if not bits:
        return _empty("◌", "JD matching is off, or no JDs are loaded for this role.")
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
            f'<td><span class="nc-tag">{s.company_archetype}</span></td>'
            f'<td class="num">{s.score_before:.0f}</td>'
            f'<td class="num">{s.score_after:.0f}</td>'
            f'<td><span class="nc-s {klass}">{s.delta:+.1f}</span></td>'
            f'<td class="gap">{gaps}</td></tr>'
        )
    table = '<div class="nc-table-wrap"><table class="nc-table">' + "".join(rows_html) + "</table></div>"

    delta_cls = "high" if report.avg_delta >= 5 else ("mid" if report.avg_delta >= 0 else "low")
    avg_block = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:14px;flex-wrap:wrap;gap:8px">'
        f'<div><b style="font-size:15px;letter-spacing:-.015em;color:var(--nc-text)">{role_pretty}</b> '
        f'<span class="nc-tag" style="margin-left:6px">{kind}</span></div>'
        f'<div style="font-size:13px;color:var(--nc-text2)">'
        f'avg: <b style="color:var(--nc-text)">{report.avg_score_before:.1f}</b> → '
        f'<b style="color:var(--nc-text)">{report.avg_score_after:.1f}</b> '
        f'<span class="nc-s {delta_cls}" style="margin-left:6px">{report.avg_delta:+.1f}</span>'
        f'</div></div>'
    )
    gaps_block = ""
    if report.top_gaps:
        gaps = "".join(f'<span class="nc-tag miss">{k}</span>' for k in report.top_gaps)
        gaps_block = (
            '<div style="margin-top:14px">'
            '<div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--nc-text3);font-weight:800;margin-bottom:8px">Top gaps</div>'
            f'{gaps}</div>'
        )
    return f'<div class="nc-card">{avg_block}{table}{gaps_block}</div>'


# ─────────────────────────────────────────────────────────────
#  Setup tab
# ─────────────────────────────────────────────────────────────
def _setup_html() -> str:
    statuses = detect_available_backends()
    instructions = {
        "huggingface": (
            "# Free tier — recommended for quick testing.\n"
            "# 1. Create a free token:\n"
            "#    https://huggingface.co/settings/tokens\n\n"
            "# 2. Set the env var:\n"
            "$env:HF_API_KEY='hf_...'      # PowerShell\n"
            "export HF_API_KEY='hf_...'    # bash/zsh\n\n"
            "# 3. Restart the app."
        ),
    }
    cards = []
    for bid, info in statuses.items():
        klass = "ready" if info["ready"] else ""
        instr = instructions.get(bid, "")
        cards.append(
            f'<div class="nc-setup-card {klass}">'
            f'<h4>{info["name"]}</h4>'
            f'<p class="why">{info["needs"]}</p>'
            f'<div class="steps">{instr}</div>'
            f'<span class="cost">{info["cost"]}</span>'
            f'</div>'
        )
    grid = '<div class="nc-setup-grid">' + "".join(cards) + "</div>"

    best = best_available_backend()
    status_card = (
        '<div class="nc-card" style="margin-top:14px;border-color:rgba(16,185,129,.35);background:rgba(16,185,129,.07)">'
        '<div class="nc-card-title" style="color:#6ee7b7">Auto-detected: '
        f'<code style="background:rgba(0,0,0,.4);padding:2px 8px;border-radius:6px;font-size:11.5px">{best}</code></div>'
        '<div class="nc-card-sub" style="color:#6ee7b7;opacity:.8">Ready to enhance. Switch to the Enhance tab.</div></div>'
        if best else
        '<div class="nc-card" style="margin-top:14px;border-color:rgba(244,63,94,.3);background:rgba(244,63,94,.07)">'
        '<div class="nc-card-title" style="color:#fca5a5">No backend configured yet</div>'
        '<div class="nc-card-sub" style="color:#fca5a5;opacity:.8">Set HF_API_KEY and restart. '
        'Get a free token at huggingface.co/settings/tokens.</div></div>'
    )
    model_note = (
        '<div class="nc-card" style="margin-top:14px">'
        '<div class="nc-card-title">Recommended models</div>'
        '<div class="nc-card-sub" style="margin-top:8px;line-height:1.8">'
        'Generation: <code style="background:rgba(124,58,237,.15);padding:1px 7px;border-radius:5px;font-size:12px">Qwen/Qwen2.5-7B-Instruct</code> (open access)<br>'
        'Extraction: <code style="background:rgba(124,58,237,.15);padding:1px 7px;border-radius:5px;font-size:12px">Qwen/Qwen2.5-7B-Instruct</code> (same model, no separate key needed)<br>'
        '<span style="color:var(--nc-text3);font-size:12px">Note: <code>meta-llama/Llama-3.1-8B-Instruct</code> and <code>Qwen2.5-3B-Instruct</code> are '
        'not available on the free serverless API — use Qwen2.5-7B-Instruct.</span>'
        '</div></div>'
    )
    return (
        '<div class="nc-card" style="margin-bottom:14px">'
        '<div class="nc-card-title">Backend Configuration</div>'
        '<div class="nc-card-sub">Configure your AI provider. The app auto-detects whatever is set in your environment. '
        'You can also override the API key and model IDs per run in the Enhance tab.</div>'
        '</div>'
        + grid + status_card + model_note
    )


# ─────────────────────────────────────────────────────────────
#  Progress HTML with pipeline stage nodes
# ─────────────────────────────────────────────────────────────
_STAGE_DEFS = [
    ("parse",        "📄", "Parse"),
    ("repair",       "🔧", "Repair"),
    ("plan",         "📐", "Plan"),
    ("enhance",      "✨", "Enhance"),
    ("render",       "🖨", "Render"),
    ("jd_match",     "🎯", "Validate"),
]
_ENHANCE_SUBSTAGES = {"enhance_plan", "section"}


def _stage_status(name: str, stages_done: set[str], current_stage: str) -> str:
    norm = "enhance" if name in ("enhance_plan",) else name
    if norm in stages_done:  return "done"
    if current_stage and (norm == current_stage or (norm == "enhance" and current_stage in _ENHANCE_SUBSTAGES)):
        return "active"
    return "idle"


def _progress_html(
    lines: list[str],
    percent: int,
    *,
    eta_s: int = 0,
    stages_done: set | None = None,
    current_stage: str = "",
) -> str:
    stages_done = stages_done or set()
    eta_txt = f"~{eta_s}s remaining" if eta_s > 0 else "finishing up…"

    nodes = []
    for i, (sid, icon, label) in enumerate(_STAGE_DEFS):
        st = _stage_status(sid, stages_done, current_stage)
        nodes.append(f'<div class="nc-stage-wrap"><div class="nc-stage-dot {st}">{icon}</div><div class="nc-stage-lbl {st}">{label}</div></div>')
        if i < len(_STAGE_DEFS) - 1:
            conn_cls = "done" if sid in stages_done else ""
            nodes.append(f'<div class="nc-stage-conn {conn_cls}"></div>')

    items = "".join(f"<div>{ln}</div>" for ln in lines[-32:])
    return (
        '<div class="nc-pipeline">'
        '<div class="nc-pipeline-hd">'
        '<span class="nc-pipeline-title">⚡ Processing your resume</span>'
        f'<span class="nc-pipeline-eta">{percent}% · {eta_txt}</span>'
        '</div>'
        f'<div class="nc-prog-bar"><div class="nc-prog-fill" style="width:{percent}%"></div></div>'
        f'<div class="nc-stages">{"".join(nodes)}</div>'
        f'<div class="nc-log">{items}</div>'
        '</div>'
    )


def _format_event(event: str, data: dict) -> str:
    if event == "stage":
        name   = data.get("name", "")
        status = data.get("status", "")
        friendly = {
            "parse":        "Reading and validating resume",
            "repair":       "Repairing missing fields",
            "complete":     "Filling required placeholders",
            "plan":         "Adapting structure for role",
            "enhance_plan": "Planning rewrite workload",
            "render":       "Generating final LaTeX",
            "jd_match":     "Validating against role JDs",
            "llm_split":    "Optimizing model routing",
        }.get(name, "Processing")
        css  = "lok" if status == "done" else "ls"
        extra = ""
        if name == "enhance_plan" and status == "done":
            extra = f' <span class="ld">(items: {data.get("total_units",0)}, mode: {data.get("mode","")})</span>'
        return f'<span class="{css}">{friendly} · {status}</span>{extra}'
    if event == "section":
        label  = data.get("label", "")
        status = data.get("status", "")
        if status == "done":
            return f'  <span class="lok">✓ section done</span> <span class="ld">{label}</span>'
        return f'  <span class="ld">↻ in progress: {label}</span>'
    if event == "review":
        return f'  <span class="ls">hiring-manager review · {data.get("status","")}</span>'
    return f'<span class="ld">{event}: {data}</span>'


# ─────────────────────────────────────────────────────────────
#  Enhance handler
# ─────────────────────────────────────────────────────────────
def _enhance_handler(
    file,
    role_id: str,
    backend: str,
    groq_api_key: str,
    hf_api_key: str,
    hf_model: str,
    hf_extraction_model: str,
    enable_critic: bool,
    enable_review: bool,
    enable_cross: bool,
    max_iter: int,
    optimization_mode: str,
):
    blank = (
        _empty("⬡", "Section traces appear here after enhancement."),
        _empty("◑", "Hiring-manager review appears here."),
        _empty("◌", "JD scores appear here."),
        None, "",
        gr.update(visible=False),
    )

    if file is None:
        yield (_empty("↑", "Upload a <b>.tex</b> resume on the left, pick a role, then click <b>Enhance</b>."), *blank)
        return

    tex_path = Path(file.name if hasattr(file, "name") else file)
    if tex_path.suffix.lower() != ".tex":
        yield (_empty("⚠", "Please upload a <b>.tex</b> file. PDF is not supported — .tex preserves all structured data."), *blank)
        return

    try:
        fsize = tex_path.stat().st_size
        if fsize > settings.max_upload_bytes:
            yield (_empty("⚠", f"File too large ({fsize//1024}KB). Max: {settings.max_upload_kb}KB."), *blank)
            return
    except Exception:
        pass

    rate_err = _check_rate_limit()
    if rate_err:
        yield (_empty("⏳", rate_err), *blank)
        return

    if groq_api_key.strip():
        os.environ["GROQ_API_KEY"] = groq_api_key.strip()
    if hf_api_key.strip():
        os.environ["HF_API_KEY"] = hf_api_key.strip()
    if hf_model.strip():
        os.environ["HF_MODEL"] = hf_model.strip()
    if hf_extraction_model.strip():
        os.environ["HF_EXTRACTION_MODEL"] = hf_extraction_model.strip()

    if backend == "auto":
        backend = "groq" if os.environ.get("GROQ_API_KEY") else "huggingface"

    if backend != "auto" and not is_backend_configured(backend):
        yield (_empty("⚠", f"Backend <b>{backend}</b> is not configured. Check the Setup tab."), *blank)
        return

    if backend == "auto" and best_available_backend() is None:
        yield (_empty("○", "No AI backend is available. Configure one via the Setup tab."), *blank)
        return

    section_budget = 140 if optimization_mode == "accuracy" else (100 if optimization_mode == "balanced" else 70)
    use_multi_llm  = optimization_mode in ("accuracy", "balanced")

    cfg = PipelineConfig(
        role_id=role_id, backend=backend,
        enable_critic=enable_critic,
        enable_role_review=enable_review,
        enable_jd_matching=True,
        enable_cross_role=enable_cross,
        max_iterations=int(max_iter),
        enable_multi_llm=use_multi_llm,
        max_section_calls=section_budget,
        optimization_mode=optimization_mode,
    )

    q: queue.Queue = queue.Queue()
    holder: dict = {"result": None, "error": None}

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

    progress_lines: list[str] = ['<span class="ls">[start]</span> initializing pipeline…']
    progress_pct  = 3
    stage_weights = {
        "parse": 10, "repair": 10, "complete": 8, "plan": 8,
        "enhance_plan": 4, "render": 15, "jd_match": 15,
    }
    stages_done: set[str] = set()
    current_stage = "parse"
    rewrite_done  = 0
    rewrite_total = 0
    start_ts      = time.monotonic()
    last_emit     = time.monotonic()

    while True:
        try:
            event, data = q.get(timeout=0.6)
        except queue.Empty:
            if time.monotonic() - last_emit > 1.2:
                elapsed = max(1, int(time.monotonic() - start_ts))
                eta     = max(0, int((elapsed / max(progress_pct, 1)) * (100 - progress_pct)))
                yield (
                    _progress_html(progress_lines, progress_pct, eta_s=eta, stages_done=stages_done, current_stage=current_stage),
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

        progress_lines.append(_format_event(event, data))
        last_emit = time.monotonic()
        elapsed   = max(1, int(time.monotonic() - start_ts))
        eta       = max(0, int((elapsed / max(progress_pct, 1)) * (100 - progress_pct)))
        yield (
            _progress_html(progress_lines, progress_pct, eta_s=eta, stages_done=stages_done, current_stage=current_stage),
            *blank[:-3], None, "", gr.update(visible=False),
        )

    if holder["error"]:
        friendly = _friendly_error(holder["error"])
        yield (
            f'<div class="nc-err"><div class="et">Enhancement could not complete</div><div class="eb">{friendly}</div></div>',
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


def _template_preview() -> str:
    p = ROOT / "app" / "render" / "template.tex.j2"
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:10000]
    except Exception:
        return "Template preview unavailable."


# ─────────────────────────────────────────────────────────────
#  Gradio app
# ─────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    role_choices = [(name, rid) for rid, name in ROLES.items()]

    with gr.Blocks(
        title="AI Resume Enhancer · Neural Canvas",
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
            radius_size="lg",
        ),
    ) as app:

        gr.HTML(_hero_html())

        # Hidden state
        critic_state = gr.State(True)
        review_state = gr.State(True)
        cross_state  = gr.State(False)
        iter_state   = gr.State(3)

        with gr.Tabs():
            # ── Enhance ──────────────────────────────────────────
            with gr.Tab("⚡ Enhance"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML(_bento_html())

                        gr.HTML(
                            '<div class="nc-card" style="margin-bottom:12px">'
                            '<div class="nc-card-title"><span class="nc-step">1</span>Upload .tex resume</div>'
                            '<div class="nc-card-sub">Output is also a .tex — paste into '
                            '<a href="https://overleaf.com" target="_blank" style="color:#c4b5fd">Overleaf</a> to compile.</div></div>'
                        )
                        file_in = gr.File(
                            label="Resume (.tex)",
                            file_types=[".tex"],
                            file_count="single",
                            height=130,
                        )

                        gr.HTML(
                            '<div class="nc-card" style="margin:12px 0">'
                            '<div class="nc-card-title"><span class="nc-step">2</span>Target role</div>'
                            '<div class="nc-card-sub">Keyword emphasis, section order, and bullet phrasing adapt to this role.</div></div>'
                        )
                        role_in = gr.Dropdown(
                            choices=role_choices, value="ai_ml_engineer",
                            label="Target role", interactive=True,
                        )

                        gr.HTML(
                            '<div class="nc-card" style="margin:12px 0">'
                            '<div class="nc-card-title"><span class="nc-step">3</span>Quality vs speed</div>'
                            '<div class="nc-card-sub">Accuracy-first runs more critic iterations per section. Speed-first skips the critic loop.</div></div>'
                        )
                        opt_mode_in = gr.Radio(
                            choices=[
                                ("Accuracy-first (recommended)", "accuracy"),
                                ("Balanced", "balanced"),
                                ("Speed-first", "speed"),
                            ],
                            value="accuracy",
                            label="Optimization mode",
                        )

                        gr.HTML(
                            '<div class="nc-card" style="margin:12px 0">'
                            '<div class="nc-card-title"><span class="nc-step">4</span>API Key &amp; Provider</div>'
                            '<div class="nc-card-sub">'
                            '<b>Groq (recommended):</b> free at '
                            '<a href="https://console.groq.com" target="_blank" style="color:#c4b5fd">console.groq.com</a>'
                            ' — no credit card needed.<br>'
                            '<b>Hugging Face:</b> free at '
                            '<a href="https://huggingface.co/settings/tokens" target="_blank" style="color:#c4b5fd">huggingface.co/settings/tokens</a>.'
                            '</div></div>'
                        )
                        groq_key_in = gr.Textbox(
                            label="Groq API key (recommended — free & fast)",
                            type="password",
                            placeholder="gsk_...",
                        )
                        backend_in = gr.Dropdown(
                            choices=[
                                ("Groq  (Llama 4 Scout — free)", "groq"),
                                ("Hugging Face  (Qwen 2.5 — free)", "huggingface"),
                                ("Auto-detect", "auto"),
                            ],
                            value="groq",
                            label="Provider",
                            interactive=True,
                        )

                        with gr.Accordion("HuggingFace key & model IDs", open=False):
                            hf_key_in = gr.Textbox(
                                label="HuggingFace API key",
                                type="password",
                                placeholder="hf_...",
                            )
                            hf_model_in = gr.Textbox(
                                label="Generation model",
                                value="Qwen/Qwen2.5-7B-Instruct",
                            )
                            hf_extract_model_in = gr.Textbox(
                                label="Extraction model",
                                value="Qwen/Qwen2.5-7B-Instruct",
                            )

                        run_btn = gr.Button(
                            "⚡  Enhance Resume",
                            variant="primary", size="lg",
                            elem_classes=["nc-cta"],
                        )

                    with gr.Column(scale=2):
                        summary_out = gr.HTML(
                            _empty("↑", "Upload a <b>.tex</b> resume on the left, choose a target role, then hit <b>Enhance</b>.")
                        )
                        gr.HTML(
                            '<div class="nc-card" style="margin-top:12px">'
                            '<div class="nc-card-title">Live pipeline progress</div>'
                            '<div class="nc-card-sub">Stage-by-stage progress, animated nodes, and an estimated wait time '
                            'will appear here while your resume is being processed.</div></div>'
                        )

            # ── Sections ─────────────────────────────────────────
            with gr.Tab("📄 Sections"):
                gr.HTML(
                    '<div class="nc-card" style="margin-bottom:14px">'
                    '<div class="nc-card-title">Per-section before / after</div>'
                    '<div class="nc-card-sub">Every rewritten section shows the original, the enhanced version, '
                    'critic iteration dots, and the final score.</div></div>'
                )
                sections_out = gr.HTML(_empty("⬡", "Section traces appear here after enhancement."))

            # ── Review ───────────────────────────────────────────
            with gr.Tab("🎯 Review"):
                gr.HTML(
                    '<div class="nc-card" style="margin-bottom:14px">'
                    '<div class="nc-card-title">Hiring-manager simulation</div>'
                    '<div class="nc-card-sub">Strengths, weaknesses, missing keywords, and a phone-screen '
                    'likelihood score for the target role.</div></div>'
                )
                review_out = gr.HTML(_empty("◑", "Review appears here after enhancement."))

            # ── JD Matching ───────────────────────────────────────
            with gr.Tab("📊 JD Matching"):
                gr.HTML(
                    '<div class="nc-card" style="margin-bottom:14px">'
                    '<div class="nc-card-title">Role job-description validation</div>'
                    '<div class="nc-card-sub">Before/after scores across curated JDs for the target role, '
                    'plus top keyword gaps to close.</div></div>'
                )
                jd_out = gr.HTML(_empty("◌", "JD scores appear here after enhancement."))

            # ── Download ─────────────────────────────────────────
            with gr.Tab("⬇ Download"):
                with gr.Group(visible=False) as download_group:
                    gr.HTML(
                        '<div class="nc-card" style="margin-bottom:12px">'
                        '<div class="nc-card-title">Enhanced .tex output</div>'
                        '<div class="nc-card-sub">Open in '
                        '<a href="https://overleaf.com" target="_blank" style="color:#c4b5fd">Overleaf</a> '
                        'to compile. Uses standard packages: '
                        '<code style="background:rgba(124,58,237,.15);padding:1px 6px;border-radius:4px;font-size:11.5px">fontawesome5</code>, '
                        '<code style="background:rgba(124,58,237,.15);padding:1px 6px;border-radius:4px;font-size:11.5px">sourcesanspro</code>, '
                        '<code style="background:rgba(124,58,237,.15);padding:1px 6px;border-radius:4px;font-size:11.5px">tabularx</code>.</div></div>'
                    )
                    tex_file_out = gr.File(label="Enhanced .tex", interactive=False)
                    tex_text_out = gr.Code(
                        label="Preview (read-only)",
                        language="latex",
                        lines=28,
                        interactive=False,
                    )

            # ── Template ─────────────────────────────────────────
            with gr.Tab("📋 Template"):
                gr.HTML(
                    '<div class="nc-card" style="margin-bottom:12px">'
                    '<div class="nc-card-title">Active Jinja2 LaTeX template</div>'
                    '<div class="nc-card-sub">This is the template used to render the final .tex. '
                    'Edit it in <code style="background:rgba(124,58,237,.15);padding:1px 6px;border-radius:4px;font-size:11.5px">app/render/template.tex.j2</code> to customize the layout.</div></div>'
                )
                gr.Code(
                    value=_template_preview(),
                    language="latex",
                    lines=28,
                    interactive=False,
                    label="template.tex.j2",
                )

            # ── Setup ─────────────────────────────────────────────
            with gr.Tab("⚙ Setup"):
                setup_html = gr.HTML(_setup_html())

        # Wire up the run button
        run_btn.click(
            _enhance_handler,
            inputs=[
                file_in, role_in, backend_in,
                groq_key_in, hf_key_in, hf_model_in, hf_extract_model_in,
                critic_state, review_state, cross_state, iter_state, opt_mode_in,
            ],
            outputs=[
                summary_out, sections_out, review_out, jd_out,
                tex_file_out, tex_text_out, download_group,
            ],
        )

        gr.HTML(
            '<div class="nc-footer">'
            'v5 Neural Canvas · Multi-agent skill-driven · '
            'HuggingFace optimized · Output: .tex (Overleaf-ready)'
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
        css=CSS,
    )
