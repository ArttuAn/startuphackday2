"""
dataset.py — PyTorch dataset for gameplay → soft emotion distribution training.

Reads from dataset-assembler/dataset/<video_id>/clips/<ts>_<emotion>/
  - gameplay.mp4  → sampled frames → preprocessed for SigLIP
  - meta.json     → soft emotion probability vector (normalised scores)

Usage:
    ds = GameplayDataset(dataset_dir, n_frames=8)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def has_face(video_path: Path, n_checks: int = 3) -> bool:
    """Return True if at least one of n_checks sampled frames contains a face."""
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False

    indices = np.linspace(0, total - 1, n_checks, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(faces) > 0:
            cap.release()
            return True

    cap.release()
    return False

# Emotion classes — must match keys produced by classify_emotions.py blendshape_to_emotions()
EMOTION_CLASSES = [
    "neutral", "happy", "excited", "sad",
    "angry", "fear", "surprise", "disgust", "contempt", "confused",
]
N_EMOTIONS = len(EMOTION_CLASSES)


def get_soft_labels(meta: dict) -> torch.Tensor:
    """
    Convert meta.json scores dict → normalised soft probability vector.

    scores values are 0-100 floats. We softmax over them so the vector
    sums to 1 and the model learns a distribution, not just a hard class.
    """
    scores = meta.get("scores", {})
    raw = torch.tensor(
        [max(0.0, float(scores.get(e, 0.0))) for e in EMOTION_CLASSES],
        dtype=torch.float32,
    )
    # If all zeros (no blendshapes detected), fall back to one-hot neutral
    if raw.sum() < 1e-6:
        raw[EMOTION_CLASSES.index("neutral")] = 1.0
    return torch.softmax(raw, dim=0)


def sample_frames_rgb(video_path: Path, n: int, img_size: int) -> np.ndarray | None:
    """
    Sample n evenly-spaced frames from video_path.
    Returns uint8 RGB array (n, img_size, img_size, 3) or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    if not frames:
        return None
    while len(frames) < n:
        frames.append(frames[-1])
    return np.stack(frames[:n], axis=0)  # (n, H, W, 3) uint8


class GameplayDataset(Dataset):
    """
    Returns (pixel_values, soft_labels) per clip.

    pixel_values: (n_frames, 3, img_size, img_size) float32, normalised for SigLIP
    soft_labels:  (N_EMOTIONS,) float32 probability distribution
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        processor,                  # SigLIP AutoProcessor
        n_frames: int = 8,
        augment: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.processor   = processor
        self.n_frames    = n_frames
        self.augment     = augment
        self.img_size    = processor.image_processor.size.get("height", 224)

        self.samples: list[tuple[Path, dict]] = []
        self._discover()

    def _discover(self):
        for vid_dir in sorted(self.dataset_dir.iterdir()):
            clips_dir = vid_dir / "clips"
            if not clips_dir.is_dir():
                continue
            for clip_dir in sorted(clips_dir.iterdir()):
                if not (clip_dir / "gameplay.mp4").exists():
                    continue
                meta_file = clip_dir / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text())
                    # Skip clips where emotion was never detected
                    if meta.get("emotion") == "none":
                        continue
                    # Skip clips where facecam has no detectable face
                    facecam = clip_dir / "facecam.mp4"
                    if facecam.exists() and not has_face(facecam):
                        continue
                    self.samples.append((clip_dir, meta))
                except Exception:
                    continue

        print(f"[GameplayDataset] {len(self.samples)} clips")
        self._print_distribution()

    def _print_distribution(self):
        from collections import Counter
        counts: Counter = Counter()
        for _, meta in self.samples:
            counts[meta.get("emotion", "none")] += 1
        for e, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {e:12s}: {c}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        clip_dir, meta = self.samples[idx]
        frames_np = sample_frames_rgb(
            clip_dir / "gameplay.mp4", self.n_frames, self.img_size)

        if frames_np is None:
            frames_np = np.zeros(
                (self.n_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        if self.augment and random.random() < 0.5:
            frames_np = frames_np[:, :, ::-1, :].copy()  # horizontal flip

        # Use SigLIP processor for each frame
        processed = self.processor(
            images=[frames_np[i] for i in range(self.n_frames)],
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values"]  # (n_frames, 3, H, W)

        soft_labels = get_soft_labels(meta)
        return pixel_values, soft_labels


def build_loaders(
    dataset_dir: Path | str,
    processor,
    val_split: float = 0.15,
    batch_size: int = 8,
    n_frames: int = 8,
    num_workers: int = 0,
    wandb_run=None,
):
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset

    full_ds = GameplayDataset(dataset_dir, processor, n_frames=n_frames, augment=True)
    if len(full_ds) == 0:
        raise RuntimeError(f"No clips found in {dataset_dir}. Run extract_clips.py first.")

    # ── Video-level split (prevents leakage: clips from same video stay together) ──
    # Group clip indices by parent video directory
    video_to_indices: dict[str, list[int]] = {}
    for i, (clip_dir, _) in enumerate(full_ds.samples):
        vid_id = clip_dir.parent.parent.name   # dataset/<vid_id>/clips/<clip>
        video_to_indices.setdefault(vid_id, []).append(i)

    video_ids = sorted(video_to_indices.keys())
    n_val_vids = max(1, int(len(video_ids) * val_split))
    # Put the last N videos in val (deterministic, no randomness needed for small sets)
    val_vids   = set(video_ids[-n_val_vids:])
    train_vids = set(video_ids[:-n_val_vids])

    train_idx = [i for vid in train_vids for i in video_to_indices[vid]]
    val_idx   = [i for vid in val_vids   for i in video_to_indices[vid]]

    # Print split info
    train_labels = [full_ds.samples[i][1].get("emotion", "none") for i in train_idx]
    val_labels   = [full_ds.samples[i][1].get("emotion", "none") for i in val_idx]
    print(f"\nTrain: {len(train_idx)} clips from {len(train_vids)} videos")
    print(f"Val:   {len(val_idx)} clips from {len(val_vids)} videos")
    print(f"Train distribution: {dict(Counter(train_labels))}")
    print(f"Val   distribution: {dict(Counter(val_labels))}\n")

    # ── Log dataset as wandb artifact ──────────────────────────────────────
    if wandb_run is not None:
        try:
            import wandb

            def _grab_frame(video_path: Path) -> np.ndarray | None:
                """Return middle frame as RGB numpy array, or None."""
                cap   = cv2.VideoCapture(str(video_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── 1. Clip summary table ──────────────────────────────────────
            rows = []
            for clip_dir, meta in full_ds.samples:
                vid_id = clip_dir.parent.parent.name
                split  = "val" if vid_id in val_vids else "train"
                rows.append([vid_id, clip_dir.name,
                              meta.get("emotion", "none"), split])

            clips_table = wandb.Table(
                columns=["video_id", "clip", "emotion", "split"],
                data=rows,
            )

            # ── 2. Class distribution bar charts ──────────────────────────
            wandb_run.log({
                "dataset/train_size":   len(train_idx),
                "dataset/val_size":     len(val_idx),
                "dataset/n_videos":     len(video_ids),
                "dataset/train_distribution": wandb.plot.bar(
                    wandb.Table(
                        columns=["emotion", "count"],
                        data=sorted(Counter(train_labels).items()),
                    ),
                    "emotion", "count", title="Train — emotion distribution",
                ),
                "dataset/val_distribution": wandb.plot.bar(
                    wandb.Table(
                        columns=["emotion", "count"],
                        data=sorted(Counter(val_labels).items()),
                    ),
                    "emotion", "count", title="Val — emotion distribution",
                ),
            })

            # ── 3. Gameplay / facecam frame pairs ─────────────────────────
            MAX_PAIRS = 30   # cap so wandb upload stays fast
            pairs_table = wandb.Table(
                columns=["split", "emotion", "gameplay_frame", "facecam_frame"])

            for split_name, indices in [("train", train_idx), ("val", val_idx)]:
                sampled = indices[:MAX_PAIRS] if len(indices) > MAX_PAIRS else indices
                for i in sampled:
                    clip_dir, meta = full_ds.samples[i]
                    emotion = meta.get("emotion", "none")

                    gp_frame = _grab_frame(clip_dir / "gameplay.mp4")
                    fc_frame = _grab_frame(clip_dir / "facecam.mp4") \
                        if (clip_dir / "facecam.mp4").exists() else None

                    gp_img = wandb.Image(gp_frame,
                                         caption=f"{split_name} | {emotion}") \
                        if gp_frame is not None else None
                    fc_img = wandb.Image(fc_frame,
                                         caption=f"{split_name} | {emotion}") \
                        if fc_frame is not None else None

                    pairs_table.add_data(split_name, emotion, gp_img, fc_img)

            wandb_run.log({
                "dataset/clips":        clips_table,
                "dataset/frame_pairs":  pairs_table,
            })

            # ── 4. Emotion inspection gallery (angry + confused) ──────────
            INSPECT_EMOTIONS = ["angry", "confused"]
            MAX_INSPECT      = 20   # max clips per emotion

            for emotion_name in INSPECT_EMOTIONS:
                inspect_table = wandb.Table(
                    columns=["clip", "video_id", "split",
                              "gameplay_frame", "facecam_frame"])
                count = 0
                for i, (clip_dir, meta) in enumerate(full_ds.samples):
                    if meta.get("emotion") != emotion_name:
                        continue
                    vid_id = clip_dir.parent.parent.name
                    split  = "val" if vid_id in val_vids else "train"

                    gp_frame = _grab_frame(clip_dir / "gameplay.mp4")
                    fc_frame = _grab_frame(clip_dir / "facecam.mp4") \
                        if (clip_dir / "facecam.mp4").exists() else None

                    inspect_table.add_data(
                        clip_dir.name,
                        vid_id,
                        split,
                        wandb.Image(gp_frame,
                                    caption=f"{emotion_name} | gameplay") \
                            if gp_frame is not None else None,
                        wandb.Image(fc_frame,
                                    caption=f"{emotion_name} | facecam") \
                            if fc_frame is not None else None,
                    )
                    count += 1
                    if count >= MAX_INSPECT:
                        break

                wandb_run.log({f"inspect/{emotion_name}": inspect_table})
                print(f"  wandb: logged {count} '{emotion_name}' clips for inspection")

            # ── 5. Save as artifact ───────────────────────────────────────
            artifact = wandb.Artifact(
                name="dataset_split", type="dataset",
                description="Train/val clip split with emotion labels and frame pairs",
                metadata={
                    "train_size": len(train_idx),
                    "val_size":   len(val_idx),
                    "train_dist": dict(Counter(train_labels)),
                    "val_dist":   dict(Counter(val_labels)),
                },
            )
            artifact.add(clips_table,  "clips")
            artifact.add(pairs_table,  "frame_pairs")
            wandb_run.log_artifact(artifact)
            print("  wandb: dataset artifact + frame pairs logged")
        except Exception as e:
            print(f"  wandb dataset logging failed: {e}")

    val_ds = GameplayDataset(dataset_dir, processor, n_frames=n_frames, augment=False)

    train_loader = DataLoader(
        Subset(full_ds, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
