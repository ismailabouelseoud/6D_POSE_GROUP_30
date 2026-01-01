# Transformer-Enhanced 6D Pose Estimation

This repository implements a modular, production-oriented pipeline for 6D object pose estimation from aligned RGB–D input. It reproduces a DenseFusion‑style architecture and experiments with a transformer‑based fusion module (custom Multi‑Head Self‑Attention) to combine global RGB and point‑cloud features.

What this code does
- Prepares and loads LINEMOD‑style RGB‑D datasets and supports YOLO‑format detections.
- Encodes RGB images and point clouds into learned feature representations.
- Fuses RGB and point‑cloud features using either a transformer‑based fuser or a baseline MLP fusion.
- Regresses 6D object pose (translation + rotation) and a per‑prediction confidence score.
- Evaluates performance with ADD / ADD‑S metrics and provides visualization utilities for qualitative inspection.

What is achieved
- A clean, importable Python package (`densefusion/`) split from the original notebook for easier review and reuse.
- A compact, from‑scratch Multi‑Head Self‑Attention and `TransformerFuser` implementation for transparent attention-based fusion.
- Reproducible training, evaluation, and visualization scripts with a simple entrypoint for quick demos.

Repository components (brief)
- `densefusion/config.py` — configuration, dataset paths, and camera intrinsics.
- `densefusion/dataset.py` — dataset loader, patch extraction, and depth→point‑cloud utilities.
- `densefusion/models.py` & `densefusion/core_models/` — encoders, fusion modules, and pose regression head.
- `densefusion/loss.py` — loss functions and ADD / ADD‑S metric helpers.
- `densefusion/segmentation.py` — Mask R‑CNN integration for segmentation masks.
- `densefusion/train.py`, `densefusion/eval.py`, `densefusion/visualize.py` — training, evaluation, and visualization utilities.
- `run_pipeline.py` — simple CLI to run train / eval flows.

See the attached report for detailed experimental settings, numeric results, and figures.

What changed in this cleanup
- Notebook → package: `densefusion/` contains modular code you can import and test.
- Added `densefusion/core_models/` with a compact, from-scratch Multi-Head Self-Attention and transformer fuser to highlight original implementation work.
- Polished `README.md`, added `requirements.txt`, and a simple entrypoint `run_pipeline.py`.

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

3. Run the training + evaluation pipeline (small smoke run possible on CPU):

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
- The report attached with this repository contains detailed experimental settings, numerical results, and figures. I recommend referencing metric tables and selected figures from the PDF in the repository README or a short `RESULTS.md` for the final presentation.

Suggested next cleanup (if you have ~4 hours)
---------------------------------------------
1. Replace external Drive-specific paths with relative dataset setup and provide a small script to download/extract a minimal subset for quick demos.
2. Add a concise `inference.py` demonstrating detection→segmentation→pose estimation using a saved model.
3. Add a short architecture diagram (can be hand-drawn and scanned) and include it in the README.

Notes & credits
---------------
- This project was refactored from the original notebook `GROUP_30_DENSEFUSION_TRS.ipynb`. Keep the notebook for reproducibility and long-form exploratory code; use the `densefusion/` package for review and code reuse.


GitHub repository containing the code for 6D pose estimation project, MLDL.  
The dataset needed to run the project can be found [here](https://drive.google.com/drive/u/0/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7).  
If you want to skip the YOLOv11 finetuning you can find its output in the **YOLOv11_finetuning** directory. The model's weights are stored in the **YOLOv11_finetuning/weights/best.pt** file.  
You can find the densefusion inspired architecture (as per code) weights for several hyperparameters combinations along with the training plots [here](https://drive.google.com/drive/folders/1C3ol0d3cCNJVmwH7Zuv1oy6GfrurOIDd?usp=sharing).  
In case you get redirected to a parent directory by clicking on the link, the above results are saved in the directory **01_MODEL_STATS**.
