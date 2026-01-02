---
# Transformer-Enhanced DenseFusion for 6D Object Pose Estimation

GitHub repository containing the code for 6D pose estimation project, MLDL.  
The dataset needed to run the project can be found [here](https://drive.google.com/drive/u/0/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)

This repository implements a Transformer‑enhanced 6D object pose estimation pipeline inspired by DenseFusion, evaluated on the LINEMOD dataset. It compares a baseline MLP fusion against a Transformer‑based fusion module that enables attention‑driven interaction between RGB and geometric features.
The full experimental report is attached as a PDF in the repository (filename:`s291365_s338570_s345149_s337049_Abouelseoud_Intini_Parisini_Ciorra.pdf`).

Overview

This code predicts 6D object poses (3D translation + 3D rotation) from aligned RGB‑D input. The pipeline is organized for reproducibility and reviewer clarity: detection → segmentation → feature extraction → fusion → pose regression.

What the code does (high level)

- Prepares LINEMOD‑style RGB‑D data and supports YOLO‑format detections.
- Encodes RGB and point‑cloud data into compact learned embeddings.
- Fuses modalities using either an MLP baseline or a Transformer‑based fuser (custom MHSA implementation included).
- Regresses object pose and a confidence score; evaluates with ADD / ADD‑S metrics.

Pipeline (brief)

1. Detection: YOLOv11 (fine‑tuned) → bounding boxes. (you can skip this part and use the finetuned models from **YOLOv11_finetuning/weights/best.pt** )
2. Segmentation: Mask R‑CNN (ResNet50) → refined masks.
3. Feature extraction: CNN (RGB) + PointNet (depth→points).
4. Fusion: MLP vs. Transformer (learnable [CLS], positional embeddings, multi‑layer encoder).
5. Pose head: MLP regresses translation, quaternion, and confidence.

Training & evaluation (summary)

- Framework: PyTorch; Optimizer: AdamW; LR: 1e‑4; Batch size: 12; Epochs: 15.
- Loss: weighted sum of ADD loss, confidence BCE, and translation regularizer:

$$L_{total} = \lambda_{ADD} L_{ADD} + \lambda_{conf} L_{conf} + 0.1 L_{reg}$$

- Metrics: ADD, ADD‑S (symmetric), success rates at 2/5/10cm and relative thresholds.

Results highlights

- Both fusion strategies achieve sub‑decimeter accuracy under the limited training regime.
- With 15 epochs the MLP baseline slightly outperformed the Transformer; this is likely due to limited compute and tuning.
- The Transformer variant trained stably and shows promise with further training and tuning.

Limitations & next steps

- Limited training time and GPU budget.
- Global, single‑descriptor fusion (no pixel‑wise refinement).
- Future: longer training, pixel‑wise fusion, evaluation on more datasets, and a pose refinement stage.

Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Update `densefusion/config.py` with your dataset and model paths (`LINEMOD_ROOT`, `PLY_MODELS_DIR`, `MODELS_SAVE_DIR`), then run:

```bash
python run_pipeline.py        # train + eval (default)
python -c "from run_pipeline import main; main('train')"   # train only
python -c "from run_pipeline import main; main('eval')"    # eval only
```

Repository layout (key files)

- `densefusion/` — core package (config, dataset, models, loss, train, eval, visualize)
- `densefusion/core_models/` — compact MHSA & Transformer fuser (from‑scratch implementation)
- `run_pipeline.py` — entrypoint to run train/eval flows
- `GROUP_30_DENSEFUSION_TRS.ipynb` — original notebook (kept for reproducibility)

Authors

- Ismail Abouelseoud
- Valeria Intini
- Gabriele Parisini
- Edoardo Ciorra

---

Repository layout (important files)
- `densefusion/`
  - `config.py` — project configuration (paths, camera intrinsics, hyperparams)
  - `utils.py` — helpers: bbox conversions, ADD/ADD-S, pose utilities
  - `models.py` — original DenseFusion-style model definitions (compatible with custom fuser)
  - `dataset.py` — `DenseFusionDataset` (rgb/depth loading, depth→pointcloud)
  - `loss.py` — `DenseFusionLoss` (ADD/ADD-S + confidence)
  - `segmentation.py` — Mask R-CNN wrapper for segmentation refinement
  - `train.py`, `eval.py`, `visualize.py` — pipeline helpers
  - `core_models/` — compact reviewer-focused models (custom MHSA & small pose estimator)
- `run_pipeline.py` — simple script to run training/eval
- `GROUP_30_DENSEFUSION_TRS.ipynb` — original notebook (kept for reproducibility)
- `requirements.txt` — minimal dependencies list

Abstract (short)
----------------
This project explores transformer-enhanced fusion for DenseFusion-style 6D pose estimation on the LINEMOD dataset. The key contribution is replacing the original concatenation/MLP fusion with a small transformer encoder that fuses RGB and point-cloud global features using a learned CLS token and positional embeddings. The repository includes utilities for dataset conversion, a segmentation refinement module (Mask R-CNN), a pose regressor, and evaluation with ADD / ADD-S metrics.

Primary contributions (what to highlight in applications)
- Custom Multi-Head Self-Attention implemented from scratch (not just `nn.Transformer`) to make attention explicit and reviewable.
- Compact TransformerFuser: uses a learnable CLS token + positional embeddings to fuse RGB and point-cloud features.
- Clean separation of concerns: dataset, model, loss, training, evaluation, visualization.

Quick reproduction steps
-----------------------
1. Create and activate a Python venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Update dataset/model paths in `densefusion/config.py` (especially `LINEMOD_ROOT`, `PLY_MODELS_DIR`, `MODELS_SAVE_DIR`).

3. Run the training + evaluation pipeline (small smoke run possible on CPU for the full training use GPU):

```bash
# Train + evaluate (default 'complete')
python run_pipeline.py

# Or run only training:
python -c "from run_pipeline import main; main('train')"

# Or run only evaluation:
python -c "from run_pipeline import main; main('eval')"
```

Smoke test for the custom transformer (quick)
-------------------------------------------
This runs a forward pass through the small `PoseEstimator` that uses the custom MHSA-based fuser. It demonstrates the transformer implementation in isolation and is suitable for showing an interviewer you implemented attention from scratch.

```bash
python - <<'PY'
from densefusion.core_models import PoseEstimator
import torch
model = PoseEstimator(use_transformer=True)
dummy_rgb = torch.randn(2, 3, 128, 128)
dummy_pts = torch.randn(2, 500, 3)
pose, conf = model(dummy_rgb, dummy_pts)
print('pose', pose.shape, 'conf', conf.shape)
PY
```

Architecture summary
--------------------
- RGB encoder: CNN → global RGB vector
- PointNet encoder: point-cloud MLP → global point vector
- Transformer fuser (custom): project both vectors into a common embedding, prepend a CLS token, add positional embeddings, apply 2-layer transformer encoder, use CLS as fused descriptor
- Pose head: small MLP that regresses translation + quaternion and a confidence score

Evaluation and metrics
----------------------
- ADD (Average Distance) and ADD-S (symmetric) are implemented in `densefusion/utils.py` and `densefusion/loss.py`. Use `eval.py` to run the built-in evaluation loop that computes per-class and global success rates.

Integration with YOLO
---------------------
- The original notebook included dataset conversion to a YOLO-format dataset and fine-tuning of a YOLOv11 model. The modular code expects a YOLO labels/images structure and can load a trained YOLO model for detection during evaluation.

How the attached report is used
------------------------------
- The report attached with this repository contains detailed experimental settings, numerical results, and figures. 



