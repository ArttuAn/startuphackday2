"""
generate.py — Retrieval-based reaction generator using SigLIP embeddings.

Pipeline:
  1. Encode gameplay windows with frozen SigLIP → predict soft emotion distribution
  2. Retrieve best-matching facecam clip by comparing soft emotion vectors
  3. Composite retrieved facecam onto gameplay video with ffmpeg

Usage:
    # Build retrieval index (run once after training)
    python generate.py --build_index

    # Generate reaction video
    python generate.py --input gameplay.mp4 --output reaction.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import (EMOTION_CLASSES, N_EMOTIONS, AUDIO_DIM,
                     get_soft_labels, sample_frames_rgb,
                     extract_gameplay_audio_features)
from train import SigLIPEmotionPredictor

BASE = Path(__file__).parent
DEFAULT_DATASET    = BASE.parent / "dataset-assembler" / "dataset"
DEFAULT_CHECKPOINT = BASE / "checkpoints" / "best_model.pt"
DEFAULT_INDEX      = BASE / "checkpoints" / "retrieval_index.pt"


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_processor(checkpoint: Path, device: torch.device):
    from transformers import AutoProcessor, AutoModel

    ckpt       = torch.load(checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "google/siglip-base-patch16-224")
    n_frames   = ckpt.get("n_frames", 4)
    audio_dim  = ckpt.get("audio_dim", AUDIO_DIM)

    processor    = AutoProcessor.from_pretrained(model_name)
    siglip_model = AutoModel.from_pretrained(model_name).vision_model
    model        = SigLIPEmotionPredictor(siglip_model, audio_dim=audio_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, processor, n_frames


# ── Soft emotion prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_emotion_vector(
    model: SigLIPEmotionPredictor,
    processor,
    video_path: Path,
    n_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Predict a soft emotion probability vector for a video clip.
    Returns (N_EMOTIONS,) tensor.
    """
    img_size = processor.image_processor.size.get("height", 224)
    frames   = sample_frames_rgb(video_path, n_frames, img_size)
    if frames is None:
        return torch.ones(N_EMOTIONS) / N_EMOTIONS  # uniform if unreadable

    processed = processor(
        images=[frames[i] for i in range(n_frames)],
        return_tensors="pt",
    )
    pixel_values   = processed["pixel_values"].unsqueeze(0).to(device)  # (1, T, 3, H, W)
    audio_features = extract_gameplay_audio_features(video_path).unsqueeze(0).to(device)
    pred = model(pixel_values, audio_features).squeeze(0).cpu()  # (N_EMOTIONS,)
    return pred


# ── Retrieval index ──────────────────────────────────────────────────────────

def build_retrieval_index(
    dataset_dir: Path,
    checkpoint: Path,
    out_index: Path,
):
    """
    Pre-compute soft emotion vectors for every facecam clip.
    Saves: {soft_vecs, paths}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, n_frames = load_model_and_processor(checkpoint, device)

    clip_dirs = [
        cd
        for vid_dir in sorted(dataset_dir.iterdir())
        if (vid_dir / "clips").is_dir()
        for cd in sorted((vid_dir / "clips").iterdir())
        if (cd / "facecam.mp4").exists() and (cd / "meta.json").exists()
    ]

    print(f"Indexing {len(clip_dirs)} facecam clips...")
    all_vecs  = []
    all_paths = []

    for i, clip_dir in enumerate(clip_dirs, 1):
        meta = json.loads((clip_dir / "meta.json").read_text())
        # Use the stored blendshape scores as the ground-truth soft vector
        soft_vec = get_soft_labels(meta)
        all_vecs.append(soft_vec)
        all_paths.append(str(clip_dir))
        if i % 50 == 0:
            print(f"  {i}/{len(clip_dirs)}")

    if not all_vecs:
        raise RuntimeError("No clips indexed — run extract_clips.py first.")

    index = {
        "soft_vecs": torch.stack(all_vecs, dim=0),  # (N, N_EMOTIONS)
        "paths":     all_paths,
        "emotion_classes": EMOTION_CLASSES,
    }
    out_index.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, out_index)
    print(f"Saved retrieval index ({len(all_paths)} clips) → {out_index}")


def retrieve_clip(query_vec: torch.Tensor, index: dict) -> Path:
    """
    Find facecam clip whose soft emotion vector is closest to query_vec.
    Uses cosine similarity on the probability vectors.
    """
    vecs  = index["soft_vecs"]   # (N, N_EMOTIONS)
    paths = index["paths"]

    q      = F.normalize(query_vec.unsqueeze(0), dim=1)
    s      = F.normalize(vecs, dim=1)
    scores = (q @ s.T).squeeze(0)   # (N,)

    best = scores.argmax().item()
    return Path(paths[best])


# ── Video compositing ────────────────────────────────────────────────────────

def composite_reaction(
    gameplay_path: Path,
    clip_dir: Path,
    out_path: Path,
    corner: str = "bottom-right",
    face_scale: float = 0.25,
) -> bool:
    facecam = clip_dir / "facecam.mp4"
    if not facecam.exists():
        print(f"  no facecam.mp4 in {clip_dir}")
        return False

    cap = cv2.VideoCapture(str(gameplay_path))
    gw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fw = int(gw * face_scale)
    fh = fw

    positions = {
        "bottom-right": (gw - fw - 10, gh - fh - 10),
        "bottom-left":  (10,            gh - fh - 10),
        "top-right":    (gw - fw - 10, 10),
        "top-left":     (10,            10),
    }
    ox, oy = positions.get(corner, positions["bottom-right"])

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(gameplay_path),
        "-stream_loop", "-1",
        "-i", str(facecam),
        "-filter_complex",
        f"[1:v]scale={fw}:{fh}[face];[0:v][face]overlay={ox}:{oy}:shortest=1[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ffmpeg failed: {result.stderr.decode()[:300]}")
        return False
    return True


# ── Full inference pipeline ──────────────────────────────────────────────────

def generate_reaction(
    gameplay_path: Path,
    out_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    index_path: Path = DEFAULT_INDEX,
    corner: str = "bottom-right",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] device={device}")

    model, processor, n_frames = load_model_and_processor(checkpoint, device)

    print(f"[generate] loading retrieval index...")
    index = torch.load(index_path, map_location="cpu")

    print(f"[generate] predicting emotion from {gameplay_path.name}...")
    pred_vec = predict_emotion_vector(model, processor, gameplay_path, n_frames, device)

    print("[generate] predicted emotion distribution:")
    for e, p in sorted(zip(EMOTION_CLASSES, pred_vec.tolist()), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {e:12s} {p:.3f}  {bar}")

    print("[generate] retrieving best-matching facecam clip...")
    clip_dir = retrieve_clip(pred_vec, index)
    print(f"[generate] retrieved: {clip_dir.name}")

    print(f"[generate] compositing → {out_path}")
    ok = composite_reaction(gameplay_path, clip_dir, out_path, corner=corner)
    if ok:
        print(f"[generate] done → {out_path}")
    return ok


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate reaction video")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--input",       type=Path)
    parser.add_argument("--output",      type=Path, default=Path("reaction_output.mp4"))
    parser.add_argument("--checkpoint",  type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index",       type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corner",      type=str,  default="bottom-right",
                        choices=["bottom-right", "bottom-left", "top-right", "top-left"])
    args = parser.parse_args()

    if args.build_index:
        build_retrieval_index(args.dataset_dir, args.checkpoint, args.index)
        return

    if args.input is None:
        parser.error("--input required")

    generate_reaction(
        gameplay_path=args.input,
        out_path=args.output,
        checkpoint=args.checkpoint,
        index_path=args.index,
        corner=args.corner,
    )


if __name__ == "__main__":
    main()
