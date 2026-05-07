<!--
  ai-related-work  —  Profile-grade README
  Author: Deepak Chaudhary  ·  github.com/deepak1212194
-->

<h1 align="center">Hi, I'm Deepak 👋</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2400&pause=900&color=1F618D&center=true&vCenter=true&width=720&lines=AI%2FML+Engineer+%E2%80%94+4%2B+years;LLMs%2C+RAG%2C+Multi-Agent+Systems%2C+Computer+Vision;Recommendation+Systems+at+Scale;Azure+ML+%E2%80%A2+NVIDIA+NIM+%E2%80%A2+CrewAI+%E2%80%A2+FAISS" alt="Typing intro" />
</p>

<p align="center">
  <a href="https://deepak1212194.github.io/ai-related-work/">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20Open%20interactive%20portfolio-4f8cff?style=for-the-badge" alt="Open interactive portfolio" />
  </a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/deepak-chaudhary-285810b7"><img src="https://img.shields.io/badge/LinkedIn-Deepak%20Chaudhary-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="mailto:deepak1212194@gmail.com"><img src="https://img.shields.io/badge/Email-deepak1212194@gmail.com-D14836?logo=gmail&logoColor=white" alt="Email" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Azure%20ML-0078D4?logo=microsoftazure&logoColor=white" alt="Azure" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT" />
  <img src="https://komarev.com/ghpvc/?username=deepak1212194&label=Repo+views&color=blueviolet&style=flat" alt="Repo views" />
  <a href="https://github.com/deepak1212194/ai-related-work/actions"><img src="https://github.com/deepak1212194/ai-related-work/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
</p>

---

I build **production AI systems end-to-end** — from data pipelines and model fine-tuning to autoscaling cloud and edge deployments. My work spans **LLMs, RAG, recommendation, semantic search, and computer vision**.

- 🧠 Multi-agent system **demonstrated live at NVIDIA GTC 2026**
- 🛰️ Co-inventor on a **granted Indian patent** for UAV-based disaster management
- 🎓 M.Tech, Computer Science — **Indian Statistical Institute (ISI), Kolkata**
- 📍 Hyderabad, India

This repo is a curated set of focused, runnable demos written from scratch, mirroring the kinds of systems I design at work.

---

## 📁 Projects

Every project is a **runnable service** — FastAPI backend, built-in browser UI, Docker-ready. Clone, `cd` in, `docker compose up`, open the URL.

| # | Project | What it shows | Stack |
|---|---|---|---|
| **01** | [**Semantic Search & Classification**](./01-rag-faq-bot/) | Embedding retrieval + **weighted-vote domain classification** with eval metrics | FastAPI · FAISS · SBERT · Pydantic · Docker |
| **02** | [**Multi-Agent Research API**](./02-multi-agent-research-crew/) | 4-agent crew with **live SSE streaming** dashboard | FastAPI · SSE · OpenAI · Docker |
| **03** | [**SBERT Training Pipeline**](./03-sbert-pair-trainer/) | 4-stage fine-tuning pipeline + **metrics dashboard** + filesystem registry | YAML configs · Pydantic · Docker |
| **04** | [**CLIP Visual Search**](./04-clip-image-text-search/) | Multimodal retrieval with **drag-drop UI** + similarity scores | FastAPI · CLIP · Docker |
| **05** | [**Edge Person Tracker + Dwell Analytics**](./05-person-tracker-mini/) | YOLO + IoU tracker with **dwell-time monitoring**, zone occupancy, heatmap over **WebSocket** | FastAPI · WebSocket · YOLOv8 · OpenCV · Docker |
| **06** | [**Resume Enhancer — Skill-Driven Agent**](./06-resume-enhancer-app/) | AI agent with **editable skill files** (skills/*.md), **ATS scoring**, hot-reload | FastAPI · Skill Files · ATS · PyMuPDF · HF/Claude · Docker |
| **07** | [**ReAct Weather Agent**](./07-react-weather-agent/) | Pure **ReAct loop** with tool calling, SSE trace streaming, offline mode | FastAPI · OpenAI · SSE · Pydantic |
| **08** | [**Energy Forecaster**](./08-energy-forecaster/) | **SARIMA vs XGBoost** 24h-ahead demand forecasting with anomaly detection | FastAPI · XGBoost · statsmodels · pandas |
| **09** | [**Multimodal Contrastive Trainer**](./09-multimodal-contrastive-trainer/) | **ResNet-50 + BERT** dual encoder with InfoNCE & focal loss, Recall@K eval | FastAPI · PyTorch · torchvision · HF |

> Architecture is consistent across projects: typed Pydantic schemas, env-driven config, singleton models warmed at startup, healthchecks, and a single-file UI per service.

---

## 🛠️ Tech I use

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=flat&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat" />
  <img src="https://img.shields.io/badge/CrewAI-FF6F00?style=flat" />
  <img src="https://img.shields.io/badge/FAISS-005571?style=flat" />
  <img src="https://img.shields.io/badge/XGBoost-006400?style=flat" />
  <img src="https://img.shields.io/badge/Azure%20ML-0078D4?style=flat&logo=microsoftazure&logoColor=white" />
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA-76B900?style=flat&logo=nvidia&logoColor=white" />
</p>

**Day-to-day:** PyTorch · Hugging Face · SentenceTransformers · LangChain · CrewAI · FAISS · Azure ML / OpenAI · AKS · Docker · NVIDIA NIM / DGX Spark · YOLO · DeepStream · XGBoost

---

## 🏗️ How these demos map to my real work

The repo shows **simplified, public versions** of patterns I've shipped in production:

```mermaid
flowchart LR
    P1[01 · Semantic Search] -.mirrors.-> D1[LLM-assisted classification<br/>over a constrained domain set]
    P2[02 · Multi-Agent Crew] -.mirrors.-> D2[Multi-agent compliance review<br/>with tool use]
    P3[03 · SBERT Pair Trainer] -.mirrors.-> D3[Sentence-encoder fine-tuning<br/>for matching / ranking]
    P4[04 · CLIP Search] -.mirrors.-> D4[Multimodal retrieval & blueprint<br/>understanding]
    P5[05 · Person Tracker] -.mirrors.-> D5[Real-time edge CV with<br/>dwell-time analytics]
    P7[07 · ReAct Agent] -.mirrors.-> D7[Tool-calling agents for<br/>domain-specific Q&A]
    P8[08 · Energy Forecaster] -.mirrors.-> D8[Time-series forecasting<br/>for operational planning]
    P9[09 · Contrastive Trainer] -.mirrors.-> D9[Multimodal encoder training<br/>with contrastive objectives]
```

| Demo here | Pattern in production |
|---|---|
| Semantic Search & Classification | LLM-assisted classification over a constrained domain set |
| Multi-Agent Research Crew | Multi-agent compliance review with tool use |
| SBERT Pair Trainer | Sentence-encoder fine-tuning for matching / ranking |
| CLIP Image-Text Search | Multimodal retrieval & document understanding |
| Person Tracker + Dwell Analytics | Real-time edge CV on RTSP camera streams with IoT alerting |
| ReAct Weather Agent | Tool-calling agents for domain-specific Q&A |
| Energy Forecaster | Time-series forecasting with model comparison |
| Multimodal Contrastive Trainer | Multimodal encoder training with contrastive objectives |

---

## 📈 Stats

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github-readme-stats.vercel.app/api?username=deepak1212194&show_icons=true&theme=tokyonight&hide_border=true&include_all_commits=true&count_private=true" />
    <source media="(prefers-color-scheme: light)" srcset="https://github-readme-stats.vercel.app/api?username=deepak1212194&show_icons=true&theme=default&hide_border=true&include_all_commits=true&count_private=true" />
    <img alt="GitHub stats" height="170" src="https://github-readme-stats.vercel.app/api?username=deepak1212194&show_icons=true&theme=default&hide_border=true&include_all_commits=true&count_private=true" />
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github-readme-stats.vercel.app/api/top-langs/?username=deepak1212194&layout=compact&theme=tokyonight&hide_border=true&langs_count=8" />
    <source media="(prefers-color-scheme: light)" srcset="https://github-readme-stats.vercel.app/api/top-langs/?username=deepak1212194&layout=compact&theme=default&hide_border=true&langs_count=8" />
    <img alt="Top languages" height="170" src="https://github-readme-stats.vercel.app/api/top-langs/?username=deepak1212194&layout=compact&theme=default&hide_border=true&langs_count=8" />
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github-readme-streak-stats.herokuapp.com?user=deepak1212194&theme=tokyonight&hide_border=true" />
    <source media="(prefers-color-scheme: light)" srcset="https://github-readme-streak-stats.herokuapp.com?user=deepak1212194&theme=default&hide_border=true" />
    <img alt="GitHub streak" height="170" src="https://github-readme-streak-stats.herokuapp.com?user=deepak1212194&theme=default&hide_border=true" />
  </picture>
</p>

---

## 🐍 Contribution graph (animated)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/deepak1212194/ai-related-work/output/snake-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/deepak1212194/ai-related-work/output/snake.svg" />
    <img alt="Contribution snake" src="https://raw.githubusercontent.com/deepak1212194/ai-related-work/output/snake.svg" />
  </picture>
</p>

> The snake animation is regenerated daily by a GitHub Action ([snake.yml](./.github/workflows/snake.yml)) and lives on the [`output`](https://github.com/deepak1212194/ai-related-work/tree/output) branch. The image link will go live a few minutes after the first workflow run.

---

## 📜 License

MIT — see [LICENSE](./LICENSE). Feel free to learn from, fork, and adapt.

<p align="center"><sub>Built and maintained by <a href="https://www.linkedin.com/in/deepak-chaudhary-285810b7">Deepak Chaudhary</a> · 2026</sub></p>
