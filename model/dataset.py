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
):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset

    full_ds = GameplayDataset(dataset_dir, processor, n_frames=n_frames, augment=True)

    if len(full_ds) == 0:
        raise RuntimeError(f"No clips found in {dataset_dir}. Run extract_clips.py first.")

    indices = list(range(len(full_ds)))
    # Stratify on hard emotion label for balanced split
    labels = [full_ds.samples[i][1].get("emotion", "neutral") for i in indices]

    from collections import Counter
    counts = Counter(labels)
    can_stratify = len(set(labels)) > 1 and min(counts.values()) >= 2

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split,
        stratify=labels if can_stratify else None,
        random_state=42,
    )

    val_ds = GameplayDataset(dataset_dir, processor, n_frames=n_frames, augment=False)

    train_loader = DataLoader(
        Subset(full_ds, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
