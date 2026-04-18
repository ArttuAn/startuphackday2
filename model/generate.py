"""
generate.py — Retrieval-based reaction generator.

Pipeline:
  1. Read gameplay video in sliding windows
  2. Predict emotion per window using trained EmotionPredictor
  3. For each window emotion, retrieve the best-matching facecam clip
     from the training index (cosine similarity of feature vectors)
  4. Composite retrieved facecam into a corner of gameplay → output video

Usage:
    # Build retrieval index from training clips (run once)
    python generate.py --build_index

    # Generate reaction video for a new gameplay video
    python generate.py --input gameplay.mp4 --output reaction_video.mp4

    # Full pipeline with custom model
    python generate.py --input gameplay.mp4 --output out.mp4 \
        --checkpoint checkpoints/best_model.pt \
        --index checkpoints/retrieval_index.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import EMOTION_CLASSES, EMOTION_TO_IDX, sample_frames
from train import EmotionPredictor

BASE = Path(__file__).parent
DEFAULT_DATASET    = BASE.parent / "dataset-assembler" / "dataset"
DEFAULT_CHECKPOINT = BASE / "checkpoints" / "best_model.pt"
DEFAULT_INDEX      = BASE / "checkpoints" / "retrieval_index.pt"


# ── Feature extraction ─────────────────────────────────────────────────────

def load_model(checkpoint: Path, device: torch.device):
    """Load EmotionPredictor from checkpoint."""
    ckpt = torch.load(checkpoint, map_location=device)
    n_classes = ckpt["n_classes"]
    model = EmotionPredictor(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_features(model: EmotionPredictor, video_path: Path,
                     n_frames: int, img_size: int,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract per-window features + predicted emotion from a video.

    Slides a window every WIN_STRIDE seconds.
    Returns (features, pred_labels) — shape (N, feat_dim) and (N,).
    """
    WIN_STRIDE = 5  # seconds between windows

    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total / fps
    starts = np.arange(0, max(1.0, duration - WIN_STRIDE), WIN_STRIDE)

    all_features = []
    all_labels   = []

    for start in starts:
        # Re-open for each window to use -ss seeking
        frames = sample_frames_at(video_path, start, WIN_STRIDE, n_frames, img_size)
        if frames is None:
            continue

        t = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)

        # Extract backbone features (before classifier head)
        B, T, C, H, W = t.shape
        x = t.view(B * T, C, H, W)
        feats = model.features(x)
        feats = model.avgpool(feats).flatten(1)       # (B*T, 1280)
        feats = feats.view(B, T, -1).mean(1)          # (B, 1280)

        logits = model.classifier(feats)              # (B, n_classes)
        pred   = logits.argmax(1).item()

        all_features.append(feats.cpu())
        all_labels.append(pred)

    if not all_features:
        return torch.zeros(0, 1280), torch.zeros(0, dtype=torch.long)

    return torch.cat(all_features, dim=0), torch.tensor(all_labels)


def sample_frames_at(video_path: Path, start: float, duration: float,
                     n: int, img_size: int) -> np.ndarray | None:
    """Sample n frames from [start, start+duration] in video_path."""
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = int(start * fps)
    f_end   = min(total - 1, int((start + duration) * fps))
    if f_end <= f_start:
        cap.release()
        return None

    indices = np.linspace(f_start, f_end, n, dtype=int)
    frames  = []
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

    arr  = np.stack(frames[:n]).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(0, 3, 1, 2)  # (n, 3, H, W)


# ── Retrieval index ────────────────────────────────────────────────────────

def build_retrieval_index(dataset_dir: Path, checkpoint: Path,
                          out_index: Path, n_frames: int = 8,
                          img_size: int = 224):
    """
    Pre-compute feature vectors for every facecam clip in the dataset.
    Saves an index: {features, labels, clip_paths}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(checkpoint, device)
    n_frames = ckpt.get("n_frames", n_frames)
    img_size = ckpt.get("img_size", img_size)

    all_features = []
    all_labels   = []
    all_paths    = []

    clip_dirs = [
        cd
        for vid_dir in sorted(dataset_dir.iterdir())
        if (vid_dir / "clips").is_dir()
        for cd in sorted((vid_dir / "clips").iterdir())
        if (cd / "facecam.mp4").exists() and (cd / "meta.json").exists()
    ]

    print(f"Building retrieval index for {len(clip_dirs)} clips...")

    for i, clip_dir in enumerate(clip_dirs, 1):
        facecam   = clip_dir / "facecam.mp4"
        meta_file = clip_dir / "meta.json"
        meta      = json.loads(meta_file.read_text())
        emotion   = meta.get("emotion", "neutral")
        label     = EMOTION_TO_IDX.get(emotion, EMOTION_TO_IDX["neutral"])

        frames = sample_frames(facecam, n_frames, img_size)
        if frames is None:
            print(f"  skip {clip_dir.name} — could not read frames")
            continue

        with torch.no_grad():
            t = torch.from_numpy(frames).unsqueeze(0).to(device)
            B, T, C, H, W = t.shape
            x     = t.view(B * T, C, H, W)
            feats = model.features(x)
            feats = model.avgpool(feats).flatten(1)
            feats = feats.view(B, T, -1).mean(1).cpu()

        all_features.append(feats)
        all_labels.append(label)
        all_paths.append(str(clip_dir))

        if i % 50 == 0:
            print(f"  {i}/{len(clip_dirs)}")

    if not all_features:
        raise RuntimeError("No clips indexed — run extract_clips.py first.")

    index = {
        "features":  torch.cat(all_features, dim=0),   # (N, 1280)
        "labels":    torch.tensor(all_labels),          # (N,)
        "paths":     all_paths,                         # list[str]
        "emotion_classes": EMOTION_CLASSES,
    }
    out_index.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, out_index)
    print(f"Saved retrieval index ({len(all_paths)} clips) → {out_index}")


def retrieve_clip(query_feat: torch.Tensor, query_label: int,
                  index: dict, top_k: int = 5) -> Path:
    """
    Find the nearest facecam clip in the index.

    Strategy:
      1. Filter index to same emotion class.
      2. Pick the clip with highest cosine similarity to query_feat.
      3. Fall back to full index if emotion class is empty.
    """
    feats  = index["features"]   # (N, D)
    labels = index["labels"]     # (N,)
    paths  = index["paths"]

    # Filter by emotion class
    mask = (labels == query_label)
    if mask.sum() == 0:
        mask = torch.ones(len(labels), dtype=torch.bool)  # fallback: all clips

    sub_feats = feats[mask]
    sub_paths = [paths[i] for i in mask.nonzero(as_tuple=True)[0].tolist()]

    q = F.normalize(query_feat.unsqueeze(0), dim=1)  # (1, D)
    s = F.normalize(sub_feats, dim=1)                 # (M, D)
    scores = (q @ s.T).squeeze(0)                     # (M,)

    best_idx = scores.topk(min(top_k, len(sub_paths))).indices[0].item()
    return Path(sub_paths[best_idx])


# ── Video compositing ──────────────────────────────────────────────────────

def composite_reaction(
    gameplay_path: Path,
    retrieved_clip_dir: Path,
    out_path: Path,
    corner: str = "bottom-right",
    face_scale: float = 0.25,
) -> bool:
    """
    Overlay facecam clip on top of gameplay video using ffmpeg.

    The facecam is placed in `corner` at `face_scale * gameplay_width`.
    Audio comes from the retrieved facecam clip (authentic streamer audio).
    """
    facecam_path = retrieved_clip_dir / "facecam.mp4"
    if not facecam_path.exists():
        print(f"  retrieved clip has no facecam.mp4: {retrieved_clip_dir}")
        return False

    # Get gameplay dimensions
    cap = cv2.VideoCapture(str(gameplay_path))
    gw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fw = int(gw * face_scale)
    fh = fw  # square facecam overlay

    if corner == "bottom-right":
        ox = gw - fw - 10
        oy = gh - fh - 10
    elif corner == "bottom-left":
        ox, oy = 10, gh - fh - 10
    elif corner == "top-right":
        ox, oy = gw - fw - 10, 10
    else:  # top-left
        ox, oy = 10, 10

    # ffmpeg: gameplay + scaled facecam overlay, facecam audio
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(gameplay_path),
        "-stream_loop", "-1",   # loop facecam if shorter than gameplay
        "-i", str(facecam_path),
        "-filter_complex",
        f"[1:v]scale={fw}:{fh}[face];"
        f"[0:v][face]overlay={ox}:{oy}:shortest=1[v]",
        "-map", "[v]",
        "-map", "1:a",           # audio from facecam clip
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ffmpeg composite failed: {result.stderr.decode()[:300]}")
        return False
    return True


# ── Full inference pipeline ────────────────────────────────────────────────

def generate_reaction(
    gameplay_path: Path,
    out_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    index_path: Path = DEFAULT_INDEX,
    n_frames: int = 8,
    img_size: int = 224,
    window_stride: float = 5.0,
    corner: str = "bottom-right",
):
    """
    Full pipeline: gameplay → predicted emotion → retrieved facecam → output video.

    For short inputs (≤ CLIP_DURATION), a single clip is retrieved and overlaid.
    For longer inputs, the video is processed window-by-window and clips are
    concatenated before compositing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] device={device}")

    model, ckpt = load_model(checkpoint, device)
    n_frames    = ckpt.get("n_frames", n_frames)
    img_size    = ckpt.get("img_size", img_size)

    print(f"[generate] loading retrieval index from {index_path}...")
    index = torch.load(index_path, map_location="cpu")

    print(f"[generate] extracting features from {gameplay_path.name}...")
    features, pred_labels = extract_features(
        model, gameplay_path, n_frames, img_size, device)

    if len(features) == 0:
        print("[generate] could not extract features — video too short?")
        return False

    print(f"[generate] {len(features)} windows, predicted emotions:")
    for i, lbl in enumerate(pred_labels.tolist()):
        print(f"  window {i}: {EMOTION_CLASSES[lbl]}")

    # Use the most common predicted emotion for retrieval
    dominant_label = pred_labels.mode().values.item()
    dominant_feat  = features[pred_labels == dominant_label].mean(0)

    print(f"[generate] dominant emotion: {EMOTION_CLASSES[dominant_label]}")
    print("[generate] retrieving best-matching facecam clip...")
    clip_dir = retrieve_clip(dominant_feat, dominant_label, index)
    print(f"[generate] retrieved: {clip_dir.name}")

    print(f"[generate] compositing → {out_path}")
    ok = composite_reaction(gameplay_path, clip_dir, out_path, corner=corner)
    if ok:
        print(f"[generate] done — {out_path}")
    return ok


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate reaction video")
    parser.add_argument("--build_index", action="store_true",
                        help="Build retrieval index from training clips")
    parser.add_argument("--input",      type=Path,
                        help="Input gameplay video path")
    parser.add_argument("--output",     type=Path, default=Path("reaction_output.mp4"),
                        help="Output composited video path")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index",      type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--dataset_dir",type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corner",     type=str,  default="bottom-right",
                        choices=["bottom-right", "bottom-left", "top-right", "top-left"])
    args = parser.parse_args()

    if args.build_index:
        build_retrieval_index(
            args.dataset_dir, args.checkpoint, args.index)
        return

    if args.input is None:
        parser.error("--input is required for reaction generation")

    generate_reaction(
        gameplay_path=args.input,
        out_path=args.output,
        checkpoint=args.checkpoint,
        index_path=args.index,
        corner=args.corner,
    )


if __name__ == "__main__":
    main()
