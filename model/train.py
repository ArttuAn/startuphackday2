"""
train.py — Train an emotion predictor on gameplay clips.

Architecture:
  - Backbone: EfficientNet-B0 (pretrained on ImageNet)
  - Temporal pooling: mean-pool frame features → single vector
  - Head: Linear(1280 → n_classes)

Training takes ~10-30 min on GPU for a small dataset.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 8 --lr 3e-4
    python train.py --dataset_dir ../dataset-assembler/dataset --out checkpoints/

Output:
    checkpoints/best_model.pt   — best validation accuracy checkpoint
    checkpoints/last_model.pt   — final epoch checkpoint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import torchvision.models as tvm
    HAS_TV = True
except ImportError:
    HAS_TV = False

from dataset import EMOTION_CLASSES, build_loaders

BASE = Path(__file__).parent
DEFAULT_DATASET = BASE.parent / "dataset-assembler" / "dataset"
DEFAULT_OUT     = BASE / "checkpoints"


# ── Model ──────────────────────────────────────────────────────────────────

class EmotionPredictor(nn.Module):
    """
    Frame-level EfficientNet-B0 backbone + temporal mean-pool + classifier.

    Input:  (batch, n_frames, 3, H, W)
    Output: (batch, n_classes) logits
    """

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        if not HAS_TV:
            raise ImportError("pip install torchvision")

        backbone = tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Remove the classifier — keep only the feature extractor
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        feat_dim       = 1280  # EfficientNet-B0 output channels

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)          # (B*T, 1280, h, w)
        x = self.avgpool(x)           # (B*T, 1280, 1, 1)
        x = x.flatten(1)              # (B*T, 1280)
        x = x.view(B, T, -1).mean(1)  # (B, 1280)  — temporal mean-pool
        return self.classifier(x)     # (B, n_classes)


# ── Training loop ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(frames)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)

    return total_loss / n, correct / n


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train gameplay emotion predictor")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out",         type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--n_frames",    type=int,   default=8,
                        help="Frames to sample per gameplay clip")
    parser.add_argument("--img_size",    type=int,   default=224)
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--val_split",   type=float, default=0.15)
    parser.add_argument("--no_amp",      action="store_true",
                        help="Disable mixed-precision training")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, n_classes = build_loaders(
        args.dataset_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        n_frames=args.n_frames,
        img_size=args.img_size,
        num_workers=args.workers,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = EmotionPredictor(n_classes=n_classes).to(device)
    print(f"Model: EmotionPredictor (EfficientNet-B0), {n_classes} classes")

    # Freeze backbone initially; fine-tune last 3 blocks after warmup
    for name, param in model.features.named_parameters():
        param.requires_grad = False

    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler     = torch.cuda.amp.GradScaler() if (
        not args.no_amp and device.type == "cuda") else None

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after 5 warmup epochs
        if epoch == 6:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=args.lr / 10,
                              weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "n_classes": n_classes,
                "emotion_classes": EMOTION_CLASSES,
                "n_frames": args.n_frames,
                "img_size": args.img_size,
            }, args.out / "best_model.pt")
            print(f"  → saved best_model.pt (val_acc={val_acc:.3f})")

    # Save last checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "val_acc": val_acc,
        "n_classes": n_classes,
        "emotion_classes": EMOTION_CLASSES,
        "n_frames": args.n_frames,
        "img_size": args.img_size,
    }, args.out / "last_model.pt")

    # Save training history
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone — best val acc: {best_val_acc:.3f}")
    print(f"Checkpoints saved to {args.out}")


if __name__ == "__main__":
    main()
