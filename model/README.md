# model/

Two-stage reaction generation pipeline.

## Architecture

```
gameplay.mp4
     │
     ▼
EmotionPredictor (EfficientNet-B0, pretrained)
  - Sample N frames per clip
  - Temporal mean-pool frame features
  - Linear classifier → emotion label
     │
     ▼
Retrieval (cosine similarity)
  - Pre-indexed facecam clips from training set
  - Filter by same emotion class
  - Pick nearest neighbour by feature similarity
     │
     ▼
ffmpeg composite
  - Overlay retrieved facecam in corner of gameplay
  - Audio from retrieved clip
     │
     ▼
reaction_output.mp4
```

## Step-by-step

### 1. Train the emotion predictor
```bash
cd model/
pip install torch torchvision scikit-learn
python train.py --epochs 20 --batch_size 8
# → checkpoints/best_model.pt
```

### 2. Build retrieval index
```bash
python generate.py --build_index
# → checkpoints/retrieval_index.pt
```

### 3. Generate a reaction video
```bash
python generate.py --input gameplay.mp4 --output reaction.mp4
```

### 4. Evaluate
```bash
# Emotion consistency + FID on held-out clips
python evaluate.py

# Cross-game generalization (point at a different game's clips)
python evaluate.py --cross_game_dir /path/to/other_game/dataset/
```

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | PyTorch Dataset — loads `clips/*/gameplay.mp4` + `meta.json` |
| `train.py` | EfficientNet-B0 training loop with warmup + fine-tuning |
| `generate.py` | Retrieval index builder + inference pipeline + ffmpeg compositing |
| `evaluate.py` | Emotion consistency accuracy, FID, cross-game distribution |

## Outputs

```
model/
  checkpoints/
    best_model.pt          ← best validation accuracy
    last_model.pt          ← final epoch
    history.json           ← loss/acc per epoch
    retrieval_index.pt     ← pre-computed facecam feature index
  eval_results/
    eval_results.json      ← emotion_consistency_accuracy, fid, cross_game_distribution
```
