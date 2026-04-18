"""
dataset.py — PyTorch dataset for gameplay → emotion training.

Reads from dataset-assembler/dataset/<video_id>/clips/<ts>_<emotion>/
  - gameplay.mp4  → sampled frames → tensor input
  - meta.json     → emotion label

Usage:
    ds = GameplayDataset(dataset_dir, n_frames=8, img_size=224)
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

# Emotions we train on (neutral/none are background noise)
EMOTION_CLASSES = [
    "happy",
    "surprised",
    "fearful",
    "disgusted",
    "angry",
    "sad",
    "neutral",
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}


def sample_frames(video_path: Path, n: int, img_size: int) -> np.ndarray | None:
    """
    Sample n evenly-spaced frames from video_path.
    Returns float32 array of shape (n, 3, img_size, img_size) or None on failure.
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

    if len(frames) < n:
        # Pad by repeating last frame
        while len(frames) < n:
            frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))

    arr = np.stack(frames[:n]).astype(np.float32) / 255.0  # (n, H, W, 3)
    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = arr.transpose(0, 3, 1, 2)  # (n, 3, H, W)
    return arr


class GameplayDataset(Dataset):
    """
    Iterates over all clip directories and returns (frames, label) pairs.

    Args:
        dataset_dir: path to dataset-assembler/dataset/
        n_frames:    frames to sample per gameplay clip
        img_size:    spatial size after resize (square)
        augment:     random horizontal flip during training
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        n_frames: int = 8,
        img_size: int = 224,
        augment: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.n_frames = n_frames
        self.img_size = img_size
        self.augment = augment

        self.samples: list[tuple[Path, int]] = []
        self._discover()

    def _discover(self):
        """Walk dataset_dir and collect (clip_dir, label_idx) pairs."""
        for vid_dir in sorted(self.dataset_dir.iterdir()):
            clips_dir = vid_dir / "clips"
            if not clips_dir.is_dir():
                continue
            for clip_dir in sorted(clips_dir.iterdir()):
                gameplay = clip_dir / "gameplay.mp4"
                meta_file = clip_dir / "meta.json"
                if not gameplay.exists() or not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text())
                    emotion = meta.get("emotion", "neutral")
                    # Map to known class; unknown → neutral
                    if emotion not in EMOTION_TO_IDX:
                        emotion = "neutral"
                    label = EMOTION_TO_IDX[emotion]
                    self.samples.append((clip_dir, label))
                except Exception:
                    continue

        print(f"[GameplayDataset] {len(self.samples)} clips found across "
              f"{len(EMOTION_CLASSES)} emotion classes")
        self._print_class_distribution()

    def _print_class_distribution(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        for idx, name in enumerate(EMOTION_CLASSES):
            print(f"  {name:12s}: {counts.get(idx, 0)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        clip_dir, label = self.samples[idx]
        gameplay = clip_dir / "gameplay.mp4"

        frames = sample_frames(gameplay, self.n_frames, self.img_size)
        if frames is None:
            # Return zeros + label on corrupt file
            frames = np.zeros((self.n_frames, 3, self.img_size, self.img_size),
                              dtype=np.float32)

        frames_tensor = torch.from_numpy(frames)  # (n, 3, H, W)

        if self.augment and random.random() < 0.5:
            frames_tensor = torch.flip(frames_tensor, dims=[-1])  # horizontal flip

        return frames_tensor, label


def build_loaders(
    dataset_dir: Path | str,
    val_split: float = 0.15,
    batch_size: int = 16,
    n_frames: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
):
    """
    Build train/val DataLoaders with a stratified split.

    Returns (train_loader, val_loader, n_classes)
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset

    full_ds = GameplayDataset(dataset_dir, n_frames=n_frames,
                              img_size=img_size, augment=True)

    if len(full_ds) == 0:
        raise RuntimeError(
            f"No clips found in {dataset_dir}. "
            "Run extract_clips.py first."
        )

    indices = list(range(len(full_ds)))
    labels  = [full_ds.samples[i][1] for i in indices]

    from collections import Counter
    label_counts = Counter(labels)
    can_stratify = len(set(labels)) > 1 and min(label_counts.values()) >= 2

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split,
        stratify=labels if can_stratify else None,
        random_state=42,
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(
        GameplayDataset(dataset_dir, n_frames=n_frames,
                        img_size=img_size, augment=False),
        val_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, len(EMOTION_CLASSES)
