# Skill: AI / ML Engineer Role

> Role-specific emphasis loaded by the agent when the user selects this role.
> Headings drive the loader: `Target Role`, `What to Emphasize`, `Priority Keywords`,
> and per-section keys (`Summary`, `Experience Bullets`, `Skills`, `Education`,
> `Achievements`) get pulled into the role profile.

---

## Target Role

AI / ML Engineer (4+ years senior-grade)

Variants this role covers:
- AI Engineer · ML Engineer · Senior AI/ML Engineer
- Applied Scientist · Generative AI Engineer · LLM Engineer
- NLP Engineer · Multimodal AI Engineer · Edge AI / Computer Vision Engineer

---

## What to Emphasize

- **End-to-end ML lifecycle ownership**: data pipelines → labeled-data
  construction → fine-tuning → held-out evaluation → autoscaling deployment.
  Senior reviewers grade heavily on ownership, not technique alone.
- **Production deployment at scale**: managed endpoints, autoscaling rules,
  multi-region rollout, observability of inference latency / cost.
- **Modern 2025-2026 GenAI stack**: LLMs, RAG, multi-agent orchestration
  (CrewAI / LangGraph / AutoGen), prompt engineering, agentic AI,
  fine-tuning approaches.
- **Vector search & retrieval**: FAISS, Pinecone, Weaviate, pgvector,
  ANN, embeddings, semantic search, cosine similarity.
- **Model evaluation rigor**: held-out testing, R² / MAE / RMSE,
  calibration, hallucination mitigation strategies.
- **Edge / GPU optimization where applicable**: NVIDIA NIM, DGX Spark,
  TensorRT, ONNX, TRTexec, CUDA shared memory, DeepStream, FP8/INT8
  quantization, Jetson deployment.
- **Public artifacts that signal seniority**: granted patents, conference
  demonstrations (NVIDIA GTC, KDD, NeurIPS workshops), public open-source
  portfolios, blog posts, papers on arXiv.
- **Cross-functional fluency**: comfortable across NLP + CV + RecSys +
  Agents — rare combinations score higher than any single specialty.

---

## Priority Keywords

LLM, RAG, fine-tuning, multi-agent, agentic AI, prompt engineering,
hallucination mitigation, CrewAI, LangChain, LangGraph, AutoGen,
PyTorch, TensorFlow, Hugging Face, SentenceTransformers, BERT, SBERT,
GPT-4, GPT-4o, Claude, Llama, Qwen, Mistral, vector search, FAISS,
Pinecone, Weaviate, pgvector, embeddings, ANN, semantic search,
cosine similarity, model evaluation, R-squared, MAE, RMSE, held-out
testing, A/B testing, MLflow, model registry, managed endpoints,
autoscaling, Azure ML, Azure OpenAI, Azure AI Search, AWS SageMaker,
GCP Vertex AI, AKS, EKS, GKE, Docker, Kubernetes, MLOps, CI/CD,
NVIDIA NIM, DGX Spark, TensorRT, ONNX, TRTexec, CUDA shared memory,
NVDEC, DeepStream, Jetson, FP8 inference, YOLO, U-Net, EfficientNet,
OC-SORT, OpenCV, FastAPI, REST APIs, Server-Sent Events, WebSockets,
event-driven, PostgreSQL, Cosmos DB, Azure Event Hub, Kafka

---

## Summary

Target shape (3-4 sentences, ~70-110 words):

1. **Sentence 1** — Senior framing + years + core domains.
   Avoid template opener. Lead with action ("Production-focused
   AI/ML Engineer with N+ years…" beats "AI/ML Engineer with N years
   of experience…").
2. **Sentence 2** — One marquee shipped system. Use a verb-led clause.
3. **Sentence 3** — One marquee public signal (GTC / patent / paper /
   open-source portfolio with URL).
4. **Optional 4** — Co-inventor / artifact mention.

Avoid:
- "Passionate about AI/ML"
- "Proven track record"
- "Cutting-edge"
- "Leveraging state-of-the-art"
- Listing every framework

Prefer:
- Specific stack ("PyTorch · Hugging Face · Azure ML")
- Specific platform name when verifiable from input
- Specific 1-line outcome with metric **only if input has one**

---

## Experience Bullets

Per-bullet template:

```
**Project Name** — *italic em-dash scope phrase if input supports one.*
[Architect-tier verb] [WHAT in plain English]. [HOW with tech keywords];
[scale / metric / outcome ONLY if from input].
```

Concrete examples to mirror:

```
Job Recommendation Service — the live matching engine behind X's user-side
job feed. Architected an event-driven service that turns each profile
update into per-field embeddings (Azure OpenAI) plus a fine-tuned SBERT
content vector, indexes the enriched profile in Azure AI Search, and
retrieves top-ranked jobs in real time. Runs on AKS with autoscaling.
```

```
ML Training Pipeline (End-to-End) — the model behind every X & Y
prediction. Owned the full model lifecycle — labelled-data construction
(N+ pairs, GPT-4 annotated), full fine-tuning of all-mpnet-base-v2 on
V100 GPU, held-out evaluation on ~M pairs (per-category R² > 0.83),
and deployment as an autoscaling Azure ML managed endpoint on T4 GPUs.
```

What earns full marks:
- Action verb at sentence-1 start
- Italic scope phrase when input supports
- 768-dim, V100, T4, FAISS, GPT-4o etc. preserved
- Multi-clause "what + how + scale" structure
- 3 lines max in 10.5pt LaTeX

What loses marks:
- "I worked on the recommender system"
- "Used various LLM techniques"
- Architecture data-flow narration ("the user data flows from API to…")
- Inventing a metric not in input

---

## Skills

Recommended skill bucket order (8 buckets max):

1. **Languages** — Python, SQL, C/C++, Bash
2. **ML & Deep Learning** — PyTorch, Hugging Face Transformers,
   SentenceTransformers, scikit-learn, NumPy, pandas, *Fine-tuning*,
   *Model Evaluation (held-out testing, R²/MAE/MSE)*
3. **LLMs & GenAI** — Azure OpenAI, GPT-4/4o, LangChain, CrewAI, RAG,
   Agentic AI, Multi-Agent Systems, Prompt Engineering, Hallucination
   Mitigation, NVIDIA NIM, Qwen, Llama
4. **Vector Search & Retrieval** — FAISS, Azure AI Search, Embeddings,
   Semantic Search, Cosine Similarity, ANN
5. **Computer Vision** — YOLO variants, U-Net, EfficientNet, OC-SORT,
   OpenCV, NVIDIA DeepStream
6. **Cloud & MLOps** — Azure ML (pipelines, registry, managed endpoints),
   Azure DevOps CI/CD, Docker, Kubernetes / AKS, Key Vault
7. **Data & Backend** — PostgreSQL, MySQL, Cosmos DB, FastAPI,
   REST APIs, SSE, WebSockets, event-driven architectures
8. **GPU & Edge** — NVIDIA DGX, Jetson, T4/V100, FP8 inference,
   TensorRT, ONNX, NVDEC

Rules for the skills section:
- 6-8 buckets max
- No bolding individual items inside a line (the bucket prefix is
  already bold; per-item bold looks shouty)
- Italic for techniques that aren't tools (*Fine-tuning*, *Model
  Evaluation*)
- Don't list tech the candidate hasn't shipped — verify against input

---

## Education

- ISI Kolkata / IIT / IIIT alumni: spell out the institution; recruiters
  outside India recognize IIT but may not recognize ISI as CS-strong
  (it's stats-strong). For ISI alumni, add a one-line `*Relevant
  coursework: Machine Learning, Deep Learning, Statistical Learning,
  Pattern Recognition, Computer Vision, …*` so a non-Indian reviewer
  knows the M.Tech actually covered modern ML.
- M.Tech dissertation: cite under Education, not Experience, even if
  the date range overlaps with a job.
- **No percentages / CGPA** for senior 4+ year resumes.

---

## Achievements

High-signal items (keep):
- Granted patents (with claim count)
- Conference selections (NVIDIA GTC, KDD, ACL, NeurIPS, CVPR)
- Top national exams (GATE 5+ years, UGC-NET in India)
- Hugging Face / arXiv / public model releases
- OSS commits to LangChain / sentence-transformers / vLLM

Low-signal items (drop at senior level):
- Class Representative
- DSA / Coursera certificates from non-prestige issuers
- Chess / cricket / sports wins
- "Best Student" / "Topper" awards

---

## Common Anti-Patterns for AI/ML Engineer

- Claiming "deployed Triton" when the candidate did benchmarking
  analysis only — use the team-credit pattern: *"Owned the analysis-and-
  benchmarking phase of a collaborative effort; the team's resulting
  deployment achieved …"*
- Listing LoRA / QLoRA when the candidate did full fine-tuning —
  these are different techniques, don't conflate
- Listing distributed training (DeepSpeed / FSDP / Accelerate) without
  evidence — interviewers will probe
- Claiming "100K+ users" or specific MAU figures when the input doesn't
  state them — round to "global crowd-work platform" or similar
  abstraction
- Stacking buzzwords ("OData-filtered vector similarity with
  Pydantic-validated structured outputs") — strip decoration, keep
  technical substance
