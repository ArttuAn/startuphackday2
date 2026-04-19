"""
generate.py — Retrieval-based reaction generator using SigLIP embeddings.

Pipeline:
  1. Encode gameplay windows with frozen SigLIP → predict soft emotion distribution
  2. Retrieve best-matching facecam clip by comparing soft emotion vectors
  3. Composite retrieved facecam onto gameplay video with ffmpeg

Usage:
    # Build retrieval index (run once after training)
    python generate.py --build_index

    # Generate single reaction video (one emotion, whole clip)
    python generate.py --input gameplay.mp4 --output reaction.mp4

    # Generate demo video (segment-by-segment, emotion label overlay)
    python generate.py --input gameplay.mp4 --output demo.mp4 --demo
    python generate.py --input gameplay.mp4 --output demo.mp4 --demo --segment_duration 10
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
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

    # Infer audio_dim from the saved head weight shape so old checkpoints
    # (trained without audio, head=[256,768]) still load correctly.
    head_w    = ckpt["model_state"].get("head.0.weight")
    if head_w is not None:
        audio_dim = max(0, head_w.shape[1] - 768)  # 768 = SigLIP embed dim
    else:
        audio_dim = ckpt.get("audio_dim", AUDIO_DIM)

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
    """Find facecam clip whose soft emotion vector is closest to query_vec."""
    vecs  = index["soft_vecs"]   # (N, N_EMOTIONS)
    paths = index["paths"]

    q      = F.normalize(query_vec.unsqueeze(0), dim=1)
    s      = F.normalize(vecs, dim=1)
    scores = (q @ s.T).squeeze(0)   # (N,)

    best = scores.argmax().item()
    return Path(paths[best])


def retrieve_clip_topk(query_vec: torch.Tensor, index: dict, k: int = 3) -> Path:
    """
    Randomly pick from top-k matches — avoids showing the exact same clip
    for every segment that predicts the same dominant emotion.
    """
    vecs  = index["soft_vecs"]
    paths = index["paths"]

    q      = F.normalize(query_vec.unsqueeze(0), dim=1)
    s      = F.normalize(vecs, dim=1)
    scores = (q @ s.T).squeeze(0)

    k       = min(k, len(paths))
    indices = scores.topk(k).indices.tolist()
    chosen  = random.choice(indices)
    return Path(paths[chosen])


# ── Video helpers ────────────────────────────────────────────────────────────

def get_video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


def get_video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# ── Video compositing ────────────────────────────────────────────────────────

def composite_reaction(
    gameplay_path: Path,
    clip_dir: Path,
    out_path: Path,
    corner: str = "bottom-right",
    face_scale: float = 0.25,
) -> bool:
    """Composite facecam onto full gameplay clip (no text overlay)."""
    facecam = clip_dir / "facecam.mp4"
    if not facecam.exists():
        print(f"  no facecam.mp4 in {clip_dir}")
        return False

    gw, gh = get_video_size(gameplay_path)
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


def burn_label_opencv(segment_path: Path, label: str, labeled_path: Path) -> bool:
    """
    Use OpenCV to burn the emotion label onto every frame of the segment.
    Writes a raw video (no audio) to labeled_path.
    Avoids any fontconfig / drawtext dependency.
    """
    cap = cv2.VideoCapture(str(segment_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(labeled_path), fourcc, fps, (w, h))

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.8, w / 640)
    thickness  = max(2, int(font_scale * 2))
    x, y       = 20, 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Shadow for readability
        cv2.putText(frame, label, (x + 2, y + 2), font, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    return labeled_path.exists()


def composite_segment_with_label(
    segment_path: Path,
    clip_dir: Path,
    out_path: Path,
    emotion_label: str,
    emotion_prob: float,
    corner: str = "bottom-right",
    face_scale: float = 0.25,
) -> bool:
    """
    Burn emotion label onto segment with OpenCV, then composite facecam
    via ffmpeg. Uses gameplay audio.
    """
    import tempfile

    facecam = clip_dir / "facecam.mp4"
    if not facecam.exists():
        print(f"  no facecam.mp4 in {clip_dir}")
        return False

    gw, gh = get_video_size(segment_path)
    fw = int(gw * face_scale)
    fh = fw

    positions = {
        "bottom-right": (gw - fw - 10, gh - fh - 10),
        "bottom-left":  (10,            gh - fh - 10),
        "top-right":    (gw - fw - 10, 10),
        "top-left":     (10,            10),
    }
    ox, oy = positions.get(corner, positions["bottom-right"])

    label = f"{emotion_label.upper()}  {emotion_prob:.0%}"

    # Step 1: burn label with OpenCV → raw mp4v video (no audio)
    tmp_labeled = out_path.parent / (out_path.stem + "_labeled_raw.mp4")
    if not burn_label_opencv(segment_path, label, tmp_labeled):
        print("  OpenCV label burn failed")
        return False

    # Step 2: composite facecam onto labeled video, mux gameplay audio
    filter_complex = (
        f"[1:v]scale={fw}:{fh}[face];"
        f"[0:v][face]overlay={ox}:{oy}:shortest=1[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_labeled),        # 0: labeled video (no audio)
        "-stream_loop", "-1",
        "-i", str(facecam),            # 1: facecam (looped)
        "-i", str(segment_path),       # 2: original segment (for audio)
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "2:a",                 # gameplay audio from original
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    try:
        tmp_labeled.unlink()
    except Exception:
        pass
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
    """Original mode: one emotion prediction for the whole clip."""
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


def generate_demo(
    gameplay_path: Path,
    out_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    index_path: Path = DEFAULT_INDEX,
    corner: str = "bottom-right",
    segment_duration: int = 8,
    top_k: int = 3,
):
    """
    Demo mode: split gameplay into segments, predict emotion per segment,
    retrieve a matching facecam clip for each, overlay emotion label, concatenate.

    This makes the reaction visibly change as the game events change.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device}")

    model, processor, n_frames = load_model_and_processor(checkpoint, device)

    print(f"[demo] loading retrieval index...")
    index = torch.load(index_path, map_location="cpu")

    total_dur = get_video_duration(gameplay_path)
    starts    = list(range(0, int(total_dur), segment_duration))
    print(f"[demo] {total_dur:.0f}s gameplay → {len(starts)} segments × {segment_duration}s")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        segment_files = []

        for i, start in enumerate(starts):
            dur = min(segment_duration, total_dur - start)
            if dur < 1.0:
                break

            print(f"\n[demo] segment {i+1}/{len(starts)}  t={start:.0f}–{start+dur:.0f}s")

            # 1. Extract segment clip (faster prediction, clean concat boundary)
            seg_in = tmp / f"seg_{i:03d}_in.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(start), "-t", str(dur),
                "-i", str(gameplay_path),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k",
                str(seg_in),
            ], check=True)

            # 2. Predict emotion from this segment
            pred_vec    = predict_emotion_vector(model, processor, seg_in, n_frames, device)
            top_idx     = pred_vec.argmax().item()
            top_emotion = EMOTION_CLASSES[top_idx]
            top_prob    = pred_vec[top_idx].item()

            # Print distribution
            print(f"  emotion → {top_emotion} ({top_prob:.0%})")
            for e, p in sorted(zip(EMOTION_CLASSES, pred_vec.tolist()), key=lambda x: -x[1])[:4]:
                bar = "█" * int(p * 20)
                print(f"    {e:12s} {p:.2f}  {bar}")

            # 3. Retrieve facecam clip (random from top-k to vary reactions)
            clip_dir = retrieve_clip_topk(pred_vec, index, k=top_k)
            print(f"  clip → {clip_dir.name}")

            # 4. Composite segment with emotion label overlay
            seg_out = tmp / f"seg_{i:03d}_out.mp4"
            ok = composite_segment_with_label(
                seg_in, clip_dir, seg_out,
                emotion_label=top_emotion,
                emotion_prob=top_prob,
                corner=corner,
            )
            if ok:
                segment_files.append(seg_out)
            else:
                print(f"  [warn] skipping segment {i+1} (composite failed)")

        if not segment_files:
            print("[demo] no segments rendered — check checkpoint and index paths")
            return False

        # 5. Concatenate all segments
        print(f"\n[demo] concatenating {len(segment_files)} segments → {out_path}")
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text(
            "\n".join(f"file '{f.as_posix()}'" for f in segment_files),
            encoding="utf-8",
        )

        result = subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(out_path),
        ], capture_output=True)

        if result.returncode != 0:
            print(f"[demo] concat failed: {result.stderr.decode()[:300]}")
            return False

        print(f"[demo] done → {out_path}")
        return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate reaction video")
    parser.add_argument("--build_index",      action="store_true",
                        help="Build retrieval index from dataset")
    parser.add_argument("--demo",             action="store_true",
                        help="Demo mode: segment-by-segment reactions with emotion label overlay")
    parser.add_argument("--input",            type=Path)
    parser.add_argument("--output",           type=Path, default=Path("reaction_output.mp4"))
    parser.add_argument("--checkpoint",       type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index",            type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--dataset_dir",      type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corner",           type=str,  default="bottom-right",
                        choices=["bottom-right", "bottom-left", "top-right", "top-left"])
    parser.add_argument("--segment_duration", type=int,  default=8,
                        help="Seconds per segment in demo mode (default: 8)")
    parser.add_argument("--top_k",            type=int,  default=3,
                        help="Randomly pick from top-k retrieved clips (default: 3)")
    args = parser.parse_args()

    if args.build_index:
        build_retrieval_index(args.dataset_dir, args.checkpoint, args.index)
        return

    if args.input is None:
        parser.error("--input required")

    if args.demo:
        generate_demo(
            gameplay_path=args.input,
            out_path=args.output,
            checkpoint=args.checkpoint,
            index_path=args.index,
            corner=args.corner,
            segment_duration=args.segment_duration,
            top_k=args.top_k,
        )
    else:
        generate_reaction(
            gameplay_path=args.input,
            out_path=args.output,
            checkpoint=args.checkpoint,
            index_path=args.index,
            corner=args.corner,
        )


if __name__ == "__main__":
    main()
