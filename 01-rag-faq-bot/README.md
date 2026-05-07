# 01 · Semantic Search & Classification Service

Embedding-based document retrieval with **weighted-vote domain classification** — exposed as a FastAPI service with a built-in search UI.

This mirrors production patterns for **domain mapping** and **talent-to-job matching**: index documents by their semantic embeddings, retrieve nearest neighbours for a query, and classify the query into a domain by aggregating category votes from retrieved evidence.

## What it does

1. **Ingest** — Loads documents from `data/` (CSV, JSON, or text), chunks them, embeds with a sentence transformer, and builds a FAISS index.
2. **Search** — Encodes a natural-language query, retrieves top-K nearest documents by cosine similarity.
3. **Classify** — Aggregates category labels from retrieved documents using score-weighted voting. Returns the predicted domain with a confidence measure.
4. **Evaluate** — Computes retrieval quality metrics (Accuracy, MRR, Recall@K) on labelled documents.

## Architecture

```
Query → SentenceTransformer → FAISS Index → Top-K Documents
                                                  ↓
                                    Weighted Vote Aggregation
                                                  ↓
                                    Predicted Category + Confidence
```

## Quick start

```bash
cd 01-rag-faq-bot
pip install -r requirements.txt

# Ingest sample documents
python -m src.ingest

# Search from CLI
python -m src.pipeline -q "real-time object detection"

# Run evaluation
python -m src.pipeline --eval

# Start the API + UI
uvicorn app.main:app --reload --port 8000
# Open http://localhost:8000
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status + index info |
| `POST` | `/api/ingest` | Rebuild index from `data/` |
| `POST` | `/api/search` | Search + classify a query |
| `GET` | `/api/stats` | Index statistics, category distribution |
| `POST` | `/api/evaluate` | Retrieval quality metrics |

### Search request

```json
{
  "query": "build recommendation systems with PyTorch",
  "top_k": 5,
  "classify": true,
  "generate_answer": false
}
```

### Search response

```json
{
  "query": "build recommendation systems with PyTorch",
  "hits": [
    {"text": "...", "score": 0.847, "category": "AI/ML Engineering", "source": "documents.csv"}
  ],
  "classification": {
    "predicted_category": "AI/ML Engineering",
    "confidence": 0.72,
    "category_scores": {"AI/ML Engineering": 0.72, "NLP/NLU": 0.18, "...": "..."}
  },
  "elapsed_ms": 12
}
```

## How it maps to production

| This demo | Production pattern |
|-----------|-------------------|
| CSV document ingestion | Azure ML pipeline ingesting job descriptions from SQL/Event Hub |
| FAISS flat index | FAISS IVF index with 500K+ documents, rebuilt nightly |
| Weighted-vote classification | Domain mapping: classify jobs into taxonomy using nearest-neighbour voting |
| Evaluation metrics (MRR) | Offline evaluation before promoting a new embedding model version |

## Stack

FastAPI · FAISS · SentenceTransformers · Pydantic · Docker
