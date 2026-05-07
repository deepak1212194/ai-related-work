# 09 · Multimodal Contrastive Trainer

**Dual-encoder contrastive learning** with ResNet-50 (image) + BERT-Tiny (text), trained to align image-text pairs in a shared embedding space. Exposed as a FastAPI service with a training dashboard.

## What it demonstrates

- **Dual-encoder architecture**: Separate encoders with shared projection heads
- **Contrastive learning**: Symmetric InfoNCE loss (used in CLIP)
- **Focal variant**: Down-weights easy negatives, emphasises hard examples
- **Transfer learning**: Frozen early layers, fine-tuned later layers
- **Retrieval evaluation**: Recall@1/5/10 and Mean Rank metrics
- **Synthetic data mode**: Works out-of-box without real images

## Architecture

```
Image ──→ ResNet-50 ──→ [Freeze] ──→ ProjectionHead ──→ L2-norm ──→ Image Embedding
                                                                         ↕
                                                                   Cosine Similarity
                                                                         ↕
Text  ──→ BERT-Tiny ──→ [Freeze] ──→ ProjectionHead ──→ L2-norm ──→ Text Embedding
```

## Quick start

```bash
cd 09-multimodal-contrastive-trainer
pip install -r requirements.txt

# Train with synthetic data
uvicorn app.main:app --reload --port 8009
# Open http://localhost:8009

# Or from CLI
python -m src.model
```

## Custom data

Create `data/metadata.json`:

```json
{
  "pairs": [
    {"image": "photo1.jpg", "text": "A cat on a mat"},
    {"image": "photo2.jpg", "text": "A dog in a park"}
  ]
}
```

Place images in `data/` alongside the JSON.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status |
| `POST` | `/api/train` | Train with configurable loss/epochs/batch/lr |
| `GET` | `/api/history` | Training loss/accuracy per epoch |
| `GET` | `/api/metrics` | Retrieval evaluation (Recall@K, Mean Rank) |

## Stack

FastAPI · PyTorch · torchvision · HuggingFace Transformers · Pydantic · Docker
