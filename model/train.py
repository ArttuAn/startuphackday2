"""
train.py — Train a soft emotion predictor on top of frozen SigLIP embeddings.

Architecture:
  - Encoder: SigLIP vision model (frozen) → 768-dim patch embeddings → mean-pool
  - Temporal pool: mean over n_frames
  - MLP head: Linear(768, 256) → GELU → Dropout → Linear(256, N_EMOTIONS) → Softmax
  - Loss: KL divergence between predicted distribution and soft target from blendshapes

Only the MLP head is trained — the SigLIP encoder stays frozen.

Usage:
    pip install transformers wandb
    python train.py --wandb_key YOUR_KEY
    python train.py --epochs 30 --batch_size 4 --wandb_key YOUR_KEY
    python train.py --no_wandb   # disable wandb
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import N_EMOTIONS, EMOTION_CLASSES, build_loaders

BASE = Path(__file__).parent
DEFAULT_DATASET = BASE.parent / "dataset-assembler" / "dataset"
DEFAULT_OUT     = BASE / "checkpoints"
DEFAULT_MODEL   = "google/siglip-base-patch16-224"


# ── Model ───────────────────────────────────────────────────────────────────

class SigLIPEmotionPredictor(nn.Module):
    """
    Frozen SigLIP encoder + trainable MLP head → soft emotion distribution.

    Input:  (batch, n_frames, 3, H, W)
    Output: (batch, N_EMOTIONS) — softmax probabilities
    """

    def __init__(self, siglip_model, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = siglip_model

        for p in self.encoder.parameters():
            p.requires_grad = False

        embed_dim = self.encoder.config.hidden_size  # 768 for base

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_EMOTIONS),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = pixel_values.shape
        x = pixel_values.view(B * T, C, H, W)

        with torch.no_grad():
            out   = self.encoder(pixel_values=x)
            feats = out.last_hidden_state.mean(dim=1)   # (B*T, 768)

        feats  = feats.view(B, T, -1).mean(dim=1)       # (B, 768)
        logits = self.head(feats)                        # (B, N_EMOTIONS)
        return torch.softmax(logits, dim=-1)


# ── Metrics ─────────────────────────────────────────────────────────────────

def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(pred.log(), target, reduction="batchmean")


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    preds:   (N, N_EMOTIONS) predicted probability distributions
    targets: (N, N_EMOTIONS) soft target distributions

    Returns top-1 acc, top-3 acc, macro F1.
    """
    pred_top1  = preds.argmax(axis=1)
    true_top1  = targets.argmax(axis=1)

    top1_acc = float((pred_top1 == true_top1).mean())

    # Top-3: true class is within predicted top-3
    top3_idx  = np.argsort(preds, axis=1)[:, -3:]
    top3_acc  = float(np.array([true_top1[i] in top3_idx[i]
                                 for i in range(len(true_top1))]).mean())

    macro_f1 = float(f1_score(true_top1, pred_top1,
                               average="macro", zero_division=0))

    return {"top1_acc": top1_acc, "top3_acc": top3_acc, "macro_f1": macro_f1}


# ── Training loop ────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool,
              wandb_run=None, epoch: int = 0, global_step: list = None):
    from tqdm import tqdm

    model.train() if train else model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    phase       = "train" if train else "val"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"  {phase} e{epoch}", leave=False, dynamic_ncols=True)
        for batch_idx, (pixel_values, soft_labels) in enumerate(pbar):
            pixel_values = pixel_values.to(device)
            soft_labels  = soft_labels.to(device)

            pred = model(pixel_values)
            loss = kl_loss(pred, soft_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Log every batch to wandb so metrics appear immediately
                if wandb_run and global_step is not None:
                    global_step[0] += 1
                    wandb_run.log({"train/batch_kl_loss": loss.item()},
                                  step=global_step[0])

            total_loss  += loss.item() * soft_labels.size(0)
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(soft_labels.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n        = sum(len(p) for p in all_preds)
    avg_loss = total_loss / n
    preds    = np.concatenate(all_preds,   axis=0)
    targets  = np.concatenate(all_targets, axis=0)
    metrics  = compute_metrics(preds, targets)
    metrics["kl_loss"] = avg_loss
    return metrics


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SigLIP emotion predictor")
    parser.add_argument("--dataset_dir", type=Path,  default=DEFAULT_DATASET)
    parser.add_argument("--out",         type=Path,  default=DEFAULT_OUT)
    parser.add_argument("--model_name",  type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n_frames",    type=int,   default=4)
    parser.add_argument("--workers",     type=int,   default=0)
    parser.add_argument("--val_split",   type=float, default=0.15)
    # wandb
    parser.add_argument("--wandb_key",   type=str,   default=None,
                        help="Weights & Biases API key")
    parser.add_argument("--wandb_project", type=str, default="streamer-reaction-model")
    parser.add_argument("--no_wandb",    action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model_name}")

    # ── wandb setup ──────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    run = None
    if use_wandb:
        try:
            import wandb
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            run = wandb.init(
                project=args.wandb_project,
                config={
                    "model_name":  args.model_name,
                    "epochs":      args.epochs,
                    "batch_size":  args.batch_size,
                    "lr":          args.lr,
                    "n_frames":    args.n_frames,
                    "n_emotions":  N_EMOTIONS,
                    "emotions":    EMOTION_CLASSES,
                    "device":      str(device),
                },
            )
            print(f"wandb run: {run.url}")
        except ImportError:
            print("wandb not installed — run: pip install wandb")
            use_wandb = False

    # ── Load SigLIP ──────────────────────────────────────────────────────
    from transformers import AutoProcessor, AutoModel
    print("Loading SigLIP...")
    processor    = AutoProcessor.from_pretrained(args.model_name)
    siglip_model = AutoModel.from_pretrained(args.model_name).vision_model

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        args.dataset_dir, processor,
        val_split=args.val_split,
        batch_size=args.batch_size,
        n_frames=args.n_frames,
        num_workers=args.workers,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── Model ────────────────────────────────────────────────────────────
    model     = SigLIPEmotionPredictor(siglip_model).to(device)
    optimizer = AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.head.parameters())
    print(f"Trainable params: {n_params:,} (head only)")

    if use_wandb and run:
        wandb.watch(model.head, log="gradients", log_freq=10)

    best_val_loss = float("inf")
    history      = []
    global_step  = [0]   # mutable so run_epoch can increment it

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, optimizer, device, train=True,
                            wandb_run=run if use_wandb else None,
                            epoch=epoch, global_step=global_step)
        val_m   = run_epoch(model, val_loader, optimizer, device, train=False,
                            epoch=epoch)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train KL={train_m['kl_loss']:.4f} top1={train_m['top1_acc']:.3f} "
            f"top3={train_m['top3_acc']:.3f} f1={train_m['macro_f1']:.3f} | "
            f"val KL={val_m['kl_loss']:.4f} top1={val_m['top1_acc']:.3f} "
            f"top3={val_m['top3_acc']:.3f} f1={val_m['macro_f1']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if use_wandb and run:
            wandb.log({
                "epoch": epoch,
                "train/kl_loss":  train_m["kl_loss"],
                "train/top1_acc": train_m["top1_acc"],
                "train/top3_acc": train_m["top3_acc"],
                "train/macro_f1": train_m["macro_f1"],
                "val/kl_loss":    val_m["kl_loss"],
                "val/top1_acc":   val_m["top1_acc"],
                "val/top3_acc":   val_m["top3_acc"],
                "val/macro_f1":   val_m["macro_f1"],
                "lr": scheduler.get_last_lr()[0],
            })

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)

        if val_m["kl_loss"] < best_val_loss:
            best_val_loss = val_m["kl_loss"]
            ckpt_path = args.out / "best_model.pt"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "val_loss":        val_m["kl_loss"],
                "val_top1_acc":    val_m["top1_acc"],
                "val_top3_acc":    val_m["top3_acc"],
                "val_macro_f1":    val_m["macro_f1"],
                "model_name":      args.model_name,
                "n_frames":        args.n_frames,
                "emotion_classes": EMOTION_CLASSES,
            }, ckpt_path)
            print(f"  → saved best_model.pt (val_KL={val_m['kl_loss']:.4f})")

            # Log checkpoint as wandb artifact
            if use_wandb and run:
                artifact = wandb.Artifact(
                    name="best_model",
                    type="model",
                    description=f"Best checkpoint at epoch {epoch}, val_KL={val_m['kl_loss']:.4f}",
                    metadata={"epoch": epoch, **{f"val_{k}": v for k, v in val_m.items()}},
                )
                artifact.add_file(str(ckpt_path))
                run.log_artifact(artifact)

    # Save final checkpoint
    last_path = args.out / "last_model.pt"
    torch.save({
        "epoch":           args.epochs,
        "model_state":     model.state_dict(),
        "model_name":      args.model_name,
        "n_frames":        args.n_frames,
        "emotion_classes": EMOTION_CLASSES,
    }, last_path)
    (args.out / "history.json").write_text(json.dumps(history, indent=2))

    # Log final artifact
    if use_wandb and run:
        artifact = wandb.Artifact(name="last_model", type="model")
        artifact.add_file(str(last_path))
        artifact.add_file(str(args.out / "history.json"))
        run.log_artifact(artifact)
        run.finish()

    print(f"\nDone — best val KL: {best_val_loss:.4f}")
    print(f"Checkpoints: {args.out}")


if __name__ == "__main__":
    main()
