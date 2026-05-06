<!--
  ai-engineering-portfolio  —  Profile README
  Author: Deepak Chaudhary
-->

<h1 align="center">
  AI Engineering Portfolio
</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2400&pause=900&color=1F618D&center=true&vCenter=true&width=720&lines=Production+AI%2FML+Engineer+%E2%80%94+4%2B+years;LLMs%2C+RAG%2C+Multi-Agent+Systems%2C+Computer+Vision;Recommendation+Systems+at+Scale;Azure+ML+%E2%80%A2+NVIDIA+NIM+%E2%80%A2+CrewAI+%E2%80%A2+FAISS" alt="Typing intro" />
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/deepak-chaudhary-285810b7"><img src="https://img.shields.io/badge/LinkedIn-Deepak%20Chaudhary-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="mailto:deepak1212194@gmail.com"><img src="https://img.shields.io/badge/Email-deepak1212194@gmail.com-D14836?logo=gmail&logoColor=white" alt="Email" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Azure%20ML-0078D4?logo=microsoftazure&logoColor=white" alt="Azure" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT" />
</p>

---

## Hi, I'm Deepak 👋

I build production AI systems end-to-end — from data pipelines and model fine-tuning to autoscaling cloud and edge deployments. My recent work spans **LLMs, RAG, recommendation, semantic search, and computer vision**, shipping at scale on Azure, GPU clusters, and edge devices.

- 🧠 Multi-agent system **selected for live demonstration at NVIDIA GTC 2026**
- 🛰️ Co-inventor on a **granted Indian patent** for UAV-based disaster management
- 🎓 M.Tech, Computer Science — **Indian Statistical Institute (ISI), Kolkata**
- 🇮🇳 Based in Hyderabad, India

This repository is a curated set of focused, runnable demos — written from scratch — that mirror the kinds of systems I design at work.

---

## 📁 Projects in this repository

| # | Project | What it demonstrates | Stack |
|---|---|---|---|
| **01** | [**RAG FAQ Bot**](./01-rag-faq-bot/) | Retrieval-Augmented Generation with **hallucination guard** via constrained-context prompting | `sentence-transformers` · `FAISS` · `OpenAI` · `Pydantic` |
| **02** | [**Multi-Agent Research Crew**](./02-multi-agent-research-crew/) | Sequentially-coordinated **multi-agent** system (planner → researcher → critic → writer) with tool use | `CrewAI` · `LangChain` · `OpenAI` |
| **03** | [**SBERT Pair Trainer**](./03-sbert-pair-trainer/) | Full **fine-tuning** of a sentence encoder with `CosineSimilarityLoss` + held-out evaluation (R² / MAE / RMSE) | `sentence-transformers` · `PyTorch` · `STS-B` |

> Each project is self-contained: clone the repo, `cd` into a project, install requirements, and run `python -m src.main` (or the project-specific entry point).

---

## 🛠️ Tech I use day-to-day

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=flat&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat" />
  <img src="https://img.shields.io/badge/CrewAI-FF6F00?style=flat" />
  <img src="https://img.shields.io/badge/FAISS-005571?style=flat" />
  <img src="https://img.shields.io/badge/Azure%20ML-0078D4?style=flat&logo=microsoftazure&logoColor=white" />
  <img src="https://img.shields.io/badge/Azure%20OpenAI-0078D4?style=flat&logo=microsoftazure&logoColor=white" />
  <img src="https://img.shields.io/badge/Azure%20AI%20Search-0078D4?style=flat&logo=microsoftazure&logoColor=white" />
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA-76B900?style=flat&logo=nvidia&logoColor=white" />
</p>

```
LLMs & GenAI       :   GPT-4 / GPT-4o, Llama, Qwen, RAG, Multi-Agent (CrewAI),
                       Prompt Engineering, Hallucination Mitigation
ML & Deep Learning :   PyTorch, Hugging Face Transformers, SentenceTransformers,
                       Fine-tuning, Model Evaluation (R² / MAE / held-out testing)
Vector & Retrieval :   FAISS, Azure AI Search, Embeddings, ANN, Cosine Similarity
Computer Vision    :   YOLOv5/v7/v8, U-Net, EfficientNet, OC-SORT, NVIDIA DeepStream
Cloud & MLOps      :   Azure ML, AKS, Azure DevOps CI/CD, Managed Endpoints, Docker
GPU & Edge         :   NVIDIA NIM, DGX Spark, Jetson Nano, T4 / V100, FP8 inference
```

---

## 🏗️ How the projects map to real systems I've shipped

```mermaid
flowchart LR
    subgraph Portfolio["This repository"]
      P1[01 · RAG FAQ Bot]
      P2[02 · Multi-Agent Crew]
      P3[03 · SBERT Pair Trainer]
    end

    subgraph Production["Patterns I use in production"]
      D1[LLM-assisted classification<br/>over a constrained domain set]
      D2[Multi-agent compliance<br/>review with tool use]
      D3[Sentence-encoder fine-tuning<br/>for matching / ranking]
    end

    P1 -.mirrors.-> D1
    P2 -.mirrors.-> D2
    P3 -.mirrors.-> D3
```

---

## 📈 Stats

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=YOUR_GITHUB_USERNAME&show_icons=true&theme=default&hide_border=true&include_all_commits=true&count_private=true" height="170" />
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=YOUR_GITHUB_USERNAME&layout=compact&theme=default&hide_border=true&langs_count=8" height="170" />
</p>

> 🛈  After you create the repo, replace `YOUR_GITHUB_USERNAME` (3 places in this README) with your real handle and the cards will render automatically.

---

## 📜 License

Released under the [MIT License](./LICENSE) — feel free to learn from, fork, and adapt.

<p align="center"><sub>Built and maintained by <a href="https://www.linkedin.com/in/deepak-chaudhary-285810b7">Deepak Chaudhary</a> · 2026</sub></p>
