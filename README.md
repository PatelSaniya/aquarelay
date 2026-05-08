# Visgrid Classification Challenge

3-class classifier for the Visgrid drone detection challenge:
**BIRD** | **DRONE** | **IRRELEVANT**

Evaluated on **Macro F1** (equal weight per class, drone misses penalised via class weights).

---

## Project Structure

```
visgrid/
├── configs/
│   └── config.yaml            # All hyperparameters
├── src/
│   ├── dataset.py             # Dataset loader (images + 2D/3D trajectories)
│   ├── features.py            # Hand-crafted motion features (for LightGBM)
│   ├── utils.py               # Focal loss, metrics, checkpointing
│   ├── eda.py                 # Exploratory analysis & plots
│   ├── train_trajectory.py    # Phase 1 — LightGBM baseline
│   ├── train_image.py         # Phase 2 — CNN + temporal image model
│   ├── train_fusion.py        # Phase 3 — Full multi-modal fusion
│   ├── predict.py             # Inference + ensemble
│   └── models/
│       ├── image_model.py     # MobileNetV3 + Temporal Transformer
│       ├── trajectory_model.py# Transformer encoder over trajectories
│       └── fusion_model.py    # Cross-modal fusion model
├── checkpoints/               # Saved model weights
├── outputs/                   # Predictions, plots
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset at:
```
ddh_visgrid_classification_challenge/
├── train/  (572 objects)
└── val/    (121 objects)
```

Or override the path in `configs/config.yaml` → `data.root`.

---

## Recommended Training Pipeline

### Step 0 — EDA
```bash
python -m src.eda
```
Generates class distribution, example frames, 3D trajectories, and feature plots in `outputs/eda/`.

---

### Phase 1 — LightGBM Trajectory Baseline (fast, strong)
```bash
python -m src.train_trajectory
```
- Extracts 50+ hand-crafted motion features from 2D and 3D trajectories
- Trains LightGBM with early stopping
- Saves `checkpoints/lgbm_trajectory.txt`
- Expected: **~0.70–0.80 Macro F1** on val

Run cross-validation too:
```bash
python -m src.train_trajectory --cv
```

---

### Phase 2 — Image Sequence Model
```bash
python -m src.train_image
```
- MobileNetV3-Small backbone (pretrained ImageNet)
- Temporal Transformer over frame sequence
- Frame differencing for motion signal
- Focal loss with class weights
- Saves `checkpoints/image_best.pt`
- Expected: **~0.65–0.75 Macro F1** on val

---

### Phase 3 — Full Fusion Model (main model)
```bash
# From scratch:
python -m src.train_fusion

# Initialise image encoder from Phase 2 (recommended):
python -m src.train_fusion --pretrain_image checkpoints/image_best.pt --freeze_image --freeze_epochs 5
```
- Image encoder + 2D trajectory transformer + 3D trajectory transformer
- Cross-modal attention fusion
- Saves `checkpoints/fusion_best.pt`
- Expected: **~0.80–0.90 Macro F1** on val

---

### Inference

```bash
# Validate on val split
python -m src.predict --model fusion --ckpt checkpoints/fusion_best.pt --split val

# Ensemble (LightGBM + Fusion) — usually best
python -m src.predict \
    --model ensemble \
    --ckpt checkpoints/fusion_best.pt \
    --lgbm_ckpt checkpoints/lgbm_trajectory.txt \
    --ensemble_weights 0.4,0.6 \
    --split val

# Test set
python -m src.predict --model ensemble --split test
```

---

## Architecture Summary

```
Images (B, T, 3, 32, 32)
    → MobileNetV3 per frame        → (B, T, 256)
    → Temporal Transformer         → (B, 256)

2D Trajectories (B, T, 3)
    → Input proj + vel features
    → Transformer Encoder          → (B, 128)

3D Trajectories (B, T, 4)
    → Input proj + vel features
    → Transformer Encoder          → (B, 128)

Cross-Modal Attention
    → Concat [img, 2d, 3d]         → (B, 512)
    → MLP Classifier               → (B, 3)
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Focal loss with DRONE weight=2.5 | Missing drones is costly |
| Temporal over sequence, not single frames | 32×32 frames are ambiguous alone |
| Frame differencing (6-channel input) | Makes wing flapping visible |
| 3D trajectory as primary signal | View-independent motion |
| Split by object_id, not frames | Prevents data leakage |
| Macro F1 not accuracy | IRRELEVANT class is small |

---

## Config

All hyperparameters live in `configs/config.yaml`. Key ones:

```yaml
classes:
  weights: [1.0, 2.5, 1.5]   # BIRD, DRONE, IRRELEVANT — higher = more penalised

training:
  loss: focal                 # focal or weighted_ce
  focal_gamma: 2.0

image_model:
  backbone: mobilenet_v3_small  # or efficientnet_b0, convnext_tiny
  temporal: transformer         # or bilstm
```
