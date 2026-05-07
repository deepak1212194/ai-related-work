"""
model.py — Multimodal Contrastive Learning Model
==================================================
Multimodal Contrastive Trainer — Core Module

Implements a dual-encoder architecture:
  - Image encoder: ResNet-50 (pretrained) → projection head
  - Text encoder: BERT-Tiny → projection head

Trained with contrastive loss to align image-text pairs in a
shared embedding space. Supports two loss functions:
  - Contrastive loss (default): symmetric InfoNCE
  - Focal loss: emphasises hard negatives

Architecture mirrors production patterns for multimodal retrieval
and document understanding.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

EMBED_DIM = 256
IMAGE_SIZE = 224
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "5"))
TEMPERATURE = 0.07


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrainMetrics:
    epoch: int
    loss: float
    contrastive_accuracy: float = 0.0


@dataclass
class EvalMetrics:
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_rank: float = 0.0


@dataclass
class TrainResult:
    status: str  # "success" | "error"
    epochs_completed: int = 0
    history: List[Dict] = field(default_factory=list)
    eval_metrics: Optional[Dict] = None
    error: str = ""


# ──────────────────────────────────────────────────────────────────────
# Projection Head
# ──────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """MLP projection from encoder space to shared embedding space."""

    def __init__(self, input_dim: int, output_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Image Encoder
# ──────────────────────────────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """ResNet-50 backbone with frozen early layers + projection head."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)

        # Freeze early layers (transfer learning)
        for param in list(backbone.parameters())[:-20]:
            param.requires_grad = False

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC
        self.projector = ProjectionHead(2048, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).squeeze(-1).squeeze(-1)
        return self.projector(feat)


# ──────────────────────────────────────────────────────────────────────
# Text Encoder
# ──────────────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    """BERT-Tiny backbone + projection head."""

    def __init__(self, embed_dim: int = EMBED_DIM, model_name: str = "prajjwal1/bert-tiny"):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Freeze early layers
        for param in list(self.bert.parameters())[:-10]:
            param.requires_grad = False

        self.projector = ProjectionHead(hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = output.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projector(cls_emb)


# ──────────────────────────────────────────────────────────────────────
# Multimodal Model
# ──────────────────────────────────────────────────────────────────────
class MultimodalModel(nn.Module):
    """
    Dual-encoder model that maps images and texts to a shared space.

    Training: contrastive loss aligns matching pairs and pushes apart
    non-matching pairs in the batch.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, temperature: float = TEMPERATURE):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)

        # Scaled cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_emb @ txt_emb.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text, img_emb, txt_emb


# ──────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────
def contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE loss (used in CLIP)."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2


def focal_contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal variant — emphasises hard negatives for better retrieval."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    ce_i2t = F.cross_entropy(logits_per_image, labels, reduction='none')
    ce_t2i = F.cross_entropy(logits_per_text, labels, reduction='none')

    pt_i2t = torch.exp(-ce_i2t)
    pt_t2i = torch.exp(-ce_t2i)

    focal_i2t = ((1 - pt_i2t) ** gamma * ce_i2t).mean()
    focal_t2i = ((1 - pt_t2i) ** gamma * ce_t2i).mean()

    return (focal_i2t + focal_t2i) / 2


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class ImageTextDataset(Dataset):
    """
    Simple image-text pair dataset.

    Expects a directory with images and a metadata.json:
    {
      "pairs": [
        {"image": "img1.jpg", "text": "A cat sitting on a mat"},
        ...
      ]
    }
    """

    def __init__(self, data_dir: Path, tokenizer=None, transform=None):
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "metadata.json"

        if meta_path.exists():
            with open(meta_path) as f:
                self.pairs = json.load(f).get("pairs", [])
        else:
            # Generate synthetic pairs for demo
            self.pairs = self._generate_synthetic_pairs()

        from transformers import AutoTokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

        if transform is None:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def _generate_synthetic_pairs(self) -> List[Dict]:
        """Generate synthetic image-text pairs for demo when no real data is provided."""
        categories = [
            ("urban skyline", "A panoramic view of a modern city skyline at sunset"),
            ("forest path", "A winding trail through a dense green forest"),
            ("ocean waves", "Powerful blue ocean waves crashing on rocky shore"),
            ("mountain peak", "Snow-capped mountain peak against clear blue sky"),
            ("desert dunes", "Golden sand dunes stretching to the horizon"),
            ("rainy street", "Wet cobblestone street reflecting city lights at night"),
            ("garden flowers", "Colorful flower garden in full bloom during spring"),
            ("snowy landscape", "Pristine white snow covering a peaceful winter landscape"),
        ]
        return [{"image": None, "text": text, "category": cat} for cat, text in categories]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        text = pair["text"]

        # Try loading real image, fall back to synthetic
        img_path = self.data_dir / pair.get("image", "") if pair.get("image") else None
        if img_path and img_path.exists():
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        else:
            # Generate a deterministic synthetic image from the text hash
            torch.manual_seed(hash(text) % (2**32))
            image = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)

        tokens = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "text": text,
        }


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────
def train_model(
    data_dir: Optional[Path] = None,
    loss_type: str = "contrastive",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> TrainResult:
    """
    Train the multimodal model.

    Args:
        data_dir: Path to image-text data directory
        loss_type: "contrastive" or "focal"
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device} | Loss: {loss_type} | Epochs: {num_epochs}")

    try:
        # Dataset
        src_dir = data_dir or (BASE_DIR / "data")
        dataset = ImageTextDataset(src_dir)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=True)

        if len(dataset) < 2:
            return TrainResult(status="error", error="Need at least 2 image-text pairs")

        # Model
        model = MultimodalModel().to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        loss_fn = focal_contrastive_loss if loss_type == "focal" else contrastive_loss

        history = []

        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits_img, logits_txt, _, _ = model(images, input_ids, attention_mask)
                loss = loss_fn(logits_img, logits_txt)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

                # Contrastive accuracy
                preds = logits_img.argmax(dim=-1)
                labels = torch.arange(len(preds), device=device)
                correct += (preds == labels).sum().item()
                total += len(preds)

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            acc = correct / total if total > 0 else 0.0

            record = {"epoch": epoch, "loss": round(avg_loss, 4), "accuracy": round(acc, 4)}
            history.append(record)
            print(f"[TRAIN] Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}, Acc: {acc:.3f}")

        # Save model
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")

        # Save history
        with open(ARTIFACTS_DIR / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Evaluation
        eval_metrics = evaluate_model(model, loader, device)

        return TrainResult(
            status="success",
            epochs_completed=num_epochs,
            history=history,
            eval_metrics=eval_metrics,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainResult(status="error", error=str(e))


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """Compute retrieval metrics: Recall@1, Recall@5, Mean Rank."""
    model.eval()
    all_img_emb = []
    all_txt_emb = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(input_ids, attention_mask)

            all_img_emb.append(img_emb.cpu())
            all_txt_emb.append(txt_emb.cpu())

    all_img_emb = torch.cat(all_img_emb)
    all_txt_emb = torch.cat(all_txt_emb)
    n = len(all_img_emb)

    # Image → Text retrieval
    sims = all_img_emb @ all_txt_emb.T
    ranks = []
    for i in range(n):
        sorted_indices = sims[i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    metrics = {
        "recall_at_1": round(float((ranks <= 1).mean()), 4),
        "recall_at_5": round(float((ranks <= 5).mean()), 4),
        "recall_at_10": round(float((ranks <= 10).mean()), 4),
        "mean_rank": round(float(ranks.mean()), 2),
        "n_samples": n,
    }

    # Save
    with open(ARTIFACTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[EVAL] R@1={metrics['recall_at_1']} R@5={metrics['recall_at_5']} MeanRank={metrics['mean_rank']}")
    return metrics


if __name__ == "__main__":
    result = train_model()
    print(f"\n[DONE] Status: {result.status}")
