# 01 · RAG FAQ Bot

A small, production-shaped **Retrieval-Augmented Generation** demo with a **hallucination guard**: the LLM is *only* allowed to answer when retrieved context contains a sufficiently strong match. Otherwise it returns a structured `IDK` response instead of fabricating.

## Why this design

Two quality issues kill RAG demos in real deployments:

1. The model **hallucinates** when retrieval returns nothing relevant.
2. The model **mixes** retrieved fact with parametric knowledge in subtle, hard-to-debug ways.

This project addresses both with **two simple guards**:

- A retrieval-score floor (`MIN_SIM = 0.45`). Below it, we return a structured "I don't know" instead of calling the LLM.
- A constrained-context prompt — the LLM is told, in the system message, to answer **only** from the provided context. Pydantic validates the output schema.

## Architecture

```
docs/sample_faqs.txt
      │
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  src/ingest  │ →  │ src/retrieve │ →  │ src/generate │ →  │ src/pipeline │
│              │    │              │    │              │    │              │
│ chunk + embed│    │  FAISS top-K │    │   LLM call   │    │  end-to-end  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
 SentenceTransformer    FAISS Index      OpenAI / stub
 (all-MiniLM-L6-v2)                      (Pydantic out)
```

## Quick start

```bash
cd 01-rag-faq-bot
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Build the index from sample docs
python -m src.ingest

# Ask a question
python -m src.pipeline --question "What is hallucination mitigation in RAG?"
```

## Project layout

```
01-rag-faq-bot/
├── src/
│   ├── ingest.py       # Module 1: chunk docs, embed, build FAISS index
│   ├── retrieve.py     # Module 2: top-K vector search with score floor
│   ├── generate.py     # Module 3: constrained-context LLM call (Pydantic out)
│   └── pipeline.py     # Module 4: ingest → retrieve → generate (CLI entry)
├── data/
│   └── sample_faqs.txt # Public, hand-written FAQ pairs (no proprietary text)
├── tests/
│   └── test_retrieve.py
└── requirements.txt
```

## Notes

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` — small, fast, CPU-friendly. Swap in `all-mpnet-base-v2` for better quality at a latency cost.
- The LLM call is wrapped in a stub when `OPENAI_API_KEY` is not set, so the demo runs offline (returns the retrieved context as the "answer" and skips generation).
- All sample FAQs in `data/` are hand-written for this demo. No proprietary corpus.
