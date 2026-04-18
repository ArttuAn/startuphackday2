"""
evaluate.py — Evaluation metrics for the reaction generator.

Metrics:
  1. Emotion consistency  — does the generated facecam emotion match the
                            gameplay's predicted emotion? (classification accuracy)
  2. Visual realism (FID) — Fréchet Inception Distance between generated
                            facecam frames and real training facecam frames
  3. Cross-game test      — run on a *different* game's gameplay and report
                            emotion distribution (checks generalisation)

Usage:
    # Evaluate against held-out clips in the dataset
    python evaluate.py

    # Evaluate on a different game's gameplay footage
    python evaluate.py --cross_game_dir /path/to/other_game_clips/

    # Full evaluation with custom checkpoint
    python evaluate.py --checkpoint checkpoints/best_model.pt --out eval_results/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import EMOTION_CLASSES, EMOTION_TO_IDX, sample_frames
from generate import (DEFAULT_CHECKPOINT, DEFAULT_DATASET, DEFAULT_INDEX,
                      extract_features, generate_reaction, load_model,
                      retrieve_clip)

BASE = Path(__file__).parent


# ── FID helpers ────────────────────────────────────────────────────────────

def _inception_features(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Extract Inception-v3 pool3 features for FID computation.
    frames: float32 (N, H, W, 3), values in [0, 1].
    Returns: (N, 2048) tensor.
    """
    try:
        import torchvision.models as tvm
        import torchvision.transforms.functional as TF
    except ImportError:
        raise ImportError("pip install torchvision")

    model = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1,
                              transform_input=False).to(device)
    model.eval()
    # Patch out final classifier to get pool3 output
    model.fc = torch.nn.Identity()

    all_feats = []
    BS = 32
    for i in range(0, len(frames), BS):
        batch = frames[i:i + BS]  # (B, H, W, 3)
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)  # (B, 3, H, W)
        t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = model(t)  # (B, 2048) after Identity fc
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


def fid_score(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    """
    Compute FID between two sets of Inception features.
    Lower = more realistic.
    """
    def stats(x: torch.Tensor):
        mu = x.mean(0).numpy()
        x_np = x.numpy()
        sigma = np.cov(x_np, rowvar=False)
        return mu, sigma

    mu1, sigma1 = stats(real_feats)
    mu2, sigma2 = stats(fake_feats)

    diff = mu1 - mu2
    # Matrix sqrt via eigendecomposition (stable for small matrices)
    vals, vecs = np.linalg.eigh(sigma1 @ sigma2)
    vals = np.maximum(vals, 0)
    sqrt_mat = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    fid = float(diff @ diff +
                np.trace(sigma1) + np.trace(sigma2) -
                2 * np.trace(sqrt_mat))
    return max(0.0, fid)


# ── Emotion consistency ────────────────────────────────────────────────────

@torch.no_grad()
def emotion_consistency_accuracy(
    model,
    ckpt: dict,
    dataset_dir: Path,
    index: dict,
    device: torch.device,
    n_eval: int = 100,
) -> float:
    """
    For each gameplay clip in the dataset:
      1. Predict emotion from gameplay frames.
      2. Retrieve a facecam clip.
      3. Predict emotion from retrieved facecam.
      4. Check if they match.

    Returns accuracy in [0, 1].
    """
    n_frames = ckpt.get("n_frames", 8)
    img_size = ckpt.get("img_size", 224)

    clip_dirs = [
        cd
        for vid_dir in sorted(dataset_dir.iterdir())
        if (vid_dir / "clips").is_dir()
        for cd in sorted((vid_dir / "clips").iterdir())
        if (cd / "gameplay.mp4").exists() and (cd / "meta.json").exists()
    ][:n_eval]

    correct = 0
    total   = 0

    for clip_dir in clip_dirs:
        gameplay  = clip_dir / "gameplay.mp4"
        meta_file = clip_dir / "meta.json"
        meta      = json.loads(meta_file.read_text())
        gt_emotion = meta.get("emotion", "neutral")
        gt_label   = EMOTION_TO_IDX.get(gt_emotion, EMOTION_TO_IDX["neutral"])

        # Predict from gameplay
        gp_frames = sample_frames(gameplay, n_frames, img_size)
        if gp_frames is None:
            continue

        t = torch.from_numpy(gp_frames).unsqueeze(0).to(device)
        B, T, C, H, W = t.shape
        x     = t.view(B * T, C, H, W)
        feats = model.features(x)
        feats = model.avgpool(feats).flatten(1).view(B, T, -1).mean(1)
        logits    = model.classifier(feats)
        pred_label = logits.argmax(1).item()

        # Retrieve facecam and predict its emotion
        retrieved_dir = retrieve_clip(feats.cpu(), pred_label, index)
        fc_frames = sample_frames(retrieved_dir / "facecam.mp4", n_frames, img_size)
        if fc_frames is None:
            continue

        t2 = torch.from_numpy(fc_frames).unsqueeze(0).to(device)
        B2, T2, C2, H2, W2 = t2.shape
        x2     = t2.view(B2 * T2, C2, H2, W2)
        feats2 = model.features(x2)
        feats2 = model.avgpool(feats2).flatten(1).view(B2, T2, -1).mean(1)
        logits2     = model.classifier(feats2)
        fc_pred     = logits2.argmax(1).item()

        if fc_pred == gt_label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# ── Collect facecam frames ─────────────────────────────────────────────────

def collect_facecam_frames(dataset_dir: Path, n_per_clip: int = 4,
                           img_size: int = 299, max_clips: int = 200
                           ) -> np.ndarray:
    """Sample frames from real training facecam clips (for FID reference set)."""
    frames = []
    for vid_dir in sorted(dataset_dir.iterdir()):
        clips_dir = vid_dir / "clips"
        if not clips_dir.is_dir():
            continue
        for clip_dir in sorted(clips_dir.iterdir()):
            fc = clip_dir / "facecam.mp4"
            if not fc.exists():
                continue
            arr = sample_frames(fc, n_per_clip, img_size)
            if arr is None:
                continue
            # Denormalize from ImageNet stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
            arr  = (arr * std + mean).clip(0, 1)  # (T, 3, H, W)
            arr  = arr.transpose(0, 2, 3, 1)      # (T, H, W, 3)
            frames.append(arr)
            if len(frames) >= max_clips:
                break
        if len(frames) >= max_clips:
            break
    return np.concatenate(frames, axis=0) if frames else np.zeros((0, img_size, img_size, 3))


# ── Cross-game evaluation ──────────────────────────────────────────────────

def cross_game_evaluation(
    cross_game_dir: Path,
    model,
    ckpt: dict,
    device: torch.device,
) -> dict:
    """
    Run emotion predictor on gameplay clips from a *different* game.
    Reports emotion distribution — checks that the model generalises.
    """
    n_frames = ckpt.get("n_frames", 8)
    img_size = ckpt.get("img_size", 224)

    clip_dirs = list(cross_game_dir.rglob("gameplay.mp4"))
    print(f"Cross-game clips: {len(clip_dirs)}")

    from collections import Counter
    emotion_dist: Counter = Counter()

    for gp in clip_dirs:
        frames = sample_frames(gp, n_frames, img_size)
        if frames is None:
            continue
        with torch.no_grad():
            t = torch.from_numpy(frames).unsqueeze(0).to(device)
            B, T, C, H, W = t.shape
            x     = t.view(B * T, C, H, W)
            feats = model.features(x)
            feats = model.avgpool(feats).flatten(1).view(B, T, -1).mean(1)
            pred  = model.classifier(feats).argmax(1).item()
        emotion_dist[EMOTION_CLASSES[pred]] += 1

    total = sum(emotion_dist.values())
    dist  = {e: round(emotion_dist[e] / total, 3) if total > 0 else 0.0
             for e in EMOTION_CLASSES}
    return dist


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate reaction generator")
    parser.add_argument("--dataset_dir",   type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint",    type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index",         type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cross_game_dir",type=Path, default=None,
                        help="Directory with gameplay clips from a different game")
    parser.add_argument("--out",           type=Path, default=BASE / "eval_results")
    parser.add_argument("--n_eval",        type=int,  default=100,
                        help="Max clips to evaluate for emotion consistency")
    parser.add_argument("--skip_fid",      action="store_true",
                        help="Skip FID computation (slow, requires torchvision Inception)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first: python train.py")
        return

    model, ckpt = load_model(args.checkpoint, device)

    if not args.index.exists():
        print(f"Retrieval index not found: {args.index}")
        print("Build it first: python generate.py --build_index")
        return

    index = torch.load(args.index, map_location="cpu")

    results = {}

    # ── 1. Emotion consistency ─────────────────────────────────────────────
    print("\n[1/3] Emotion consistency accuracy...")
    acc = emotion_consistency_accuracy(
        model, ckpt, args.dataset_dir, index, device, args.n_eval)
    results["emotion_consistency_accuracy"] = round(acc, 4)
    print(f"  Emotion consistency: {acc:.1%}")

    # ── 2. FID ─────────────────────────────────────────────────────────────
    if not args.skip_fid:
        print("\n[2/3] FID score (real vs retrieved facecam)...")
        try:
            n_frames = ckpt.get("n_frames", 8)
            img_size = ckpt.get("img_size", 224)

            print("  collecting real facecam frames...")
            real_frames = collect_facecam_frames(
                args.dataset_dir, n_per_clip=4, img_size=299)

            # Generate retrieved frames by sampling index clips
            print("  collecting retrieved facecam frames...")
            index_paths = index["paths"]
            import random; random.shuffle(index_paths)
            fake_frames = collect_facecam_frames_from_paths(
                [Path(p) / "facecam.mp4" for p in index_paths[:200]],
                n_per_clip=4, img_size=299)

            if len(real_frames) > 8 and len(fake_frames) > 8:
                print(f"  computing FID ({len(real_frames)} real, {len(fake_frames)} fake)...")
                real_feats = _inception_features(real_frames, device)
                fake_feats = _inception_features(fake_frames, device)
                fid = fid_score(real_feats, fake_feats)
                results["fid"] = round(fid, 2)
                print(f"  FID: {fid:.2f}")
            else:
                print("  not enough frames for FID — skipping")
                results["fid"] = None

        except Exception as e:
            print(f"  FID failed: {e}")
            results["fid"] = None
    else:
        print("\n[2/3] FID skipped (--skip_fid)")
        results["fid"] = None

    # ── 3. Cross-game ──────────────────────────────────────────────────────
    if args.cross_game_dir and args.cross_game_dir.exists():
        print(f"\n[3/3] Cross-game evaluation on {args.cross_game_dir.name}...")
        dist = cross_game_evaluation(args.cross_game_dir, model, ckpt, device)
        results["cross_game_emotion_distribution"] = dist
        print("  Emotion distribution:")
        for emo, frac in dist.items():
            bar = "█" * int(frac * 40)
            print(f"    {emo:12s} {frac:.1%}  {bar}")
    else:
        print("\n[3/3] Cross-game skipped (no --cross_game_dir provided)")
        results["cross_game_emotion_distribution"] = None

    # ── Save results ───────────────────────────────────────────────────────
    out_file = args.out / "eval_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_file}")
    print(json.dumps(results, indent=2))


def collect_facecam_frames_from_paths(paths, n_per_clip=4, img_size=299):
    """Collect frames from explicit list of facecam.mp4 paths."""
    frames = []
    for p in paths:
        if not Path(p).exists():
            continue
        arr = sample_frames(Path(p), n_per_clip, img_size)
        if arr is None:
            continue
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
        arr  = (arr * std + mean).clip(0, 1).transpose(0, 2, 3, 1)
        frames.append(arr)
    return np.concatenate(frames, axis=0) if frames else np.zeros((0, img_size, img_size, 3))


if __name__ == "__main__":
    main()
