# Transformer-Enhanced DenseFusion for 6D Object Pose Estimation

## Overview

This repository implements a **Transformer-enhanced 6D object pose estimation** pipeline based on DenseFusion, evaluated on the LINEMOD dataset. The project compares a baseline MLP fusion architecture against a novel Transformer-based fusion module that leverages multi-head self-attention to enable fine-grained interaction between RGB and geometric features.

### Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for dataset handling, model components, loss computation, training, and evaluation
- **Dual Fusion Strategies**: Implements both MLP baseline and custom Transformer-based fusion (from-scratch implementation for full transparency)
- **End-to-End Pipeline**: Complete workflow from detection through segmentation, feature extraction, fusion, and pose regression
- **Comprehensive Evaluation**: ADD/ADD-S metrics with success rates at multiple thresholds (2cm, 5cm, 10cm)

### What This Code Does

The pipeline predicts **6D object poses** (3D translation + 3D rotation) from aligned RGB-D input:

1. **Object Detection**: YOLOv11 (fine-tuned) generates bounding boxes
2. **Segmentation Refinement**: Mask R-CNN (ResNet50) refines object masks
3. **Feature Extraction**: 
   - CNN encodes RGB patches into compact embeddings
   - PointNet encodes point clouds derived from depth data
4. **Feature Fusion**: Either MLP concatenation or attention-based Transformer fusion
5. **Pose Regression**: Final MLP head predicts 6D pose and confidence score

### Dataset

The LINEMOD dataset required to run this project is available [here](https://drive.google.com/drive/u/0/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7). Please download and extract it before running the pipeline.

For detailed experimental settings, results, and analysis, see the full report in the report directory

---

## Technical Details

### Pipeline Architecture

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| Detection | YOLOv11 | RGB image | Bounding boxes |
| Segmentation | Mask R-CNN | RGB + bounding box | Object mask |
| RGB Encoding | CNN | RGB patch | RGB embedding |
| Geometric Encoding | PointNet | Point cloud | Geometry embedding |
| Fusion | MLP or Transformer | RGB + geometry embeddings | Fused descriptor |
| Pose Regression | MLP head | Fused descriptor | Translation + Quaternion |

### Training Configuration

- **Framework**: PyTorch with mixed precision
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 12
- **Epochs**: 15
- **Loss Function**: Weighted combination of ADD loss, confidence BCE, and translation regularization:

$$L_{total} = \lambda_{ADD} L_{ADD} + \lambda_{conf} L_{conf} + 0.1 L_{reg}$$

### Evaluation Metrics

- **ADD (Average Distance)**: Mean distance between corresponding model points
- **ADD-S (Symmetric ADD)**: For symmetric objects, uses closest point distance
- **Success Rate**: Percentage of predictions within threshold distances (2cm, 5cm, 10cm)
- **Relative Thresholds**: Percentage of model diameter (5%, 10%, 20%)

---

## Results

### Performance Summary

Both fusion strategies achieved sub-decimeter accuracy under the limited training regime (15 epochs):

- **MLP Baseline**: Slightly outperformed the Transformer variant with current hyperparameters
- **Transformer Fusion**: Trained stably and demonstrates promising results with further tuning and training time

![Training results comparison: Transformer vs MLP](docs/results_table.png)
*Figure: Training results table comparing Transformer and MLP fusion strategies*

### Key Observations

- **Limited Compute Impact**: The MLP baseline's advantage is primarily attributed to limited GPU budget and training time
- **Transformer Potential**: The Transformer variant shows stable convergence and is expected to benefit from extended training
- **Sub-decimeter Accuracy**: Both methods achieve competitive accuracy for 6D pose estimation tasks

### Limitations & Future Work

**Current Limitations:**
- Limited training time and GPU budget (15 epochs on constrained hardware)
- Global, single-descriptor fusion without pixel-wise refinement
- Evaluation limited to LINEMOD dataset

**Future Directions:**
- Extended training schedule with increased computational resources
- Pixel-wise fusion for more detailed spatial reasoning
- Multi-dataset evaluation (YCB, YCBV, etc.)
- Pose refinement stage for improved accuracy
- Hyperparameter optimization specific to Transformer architecture

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (recommended for GPU acceleration)
- 10GB+ disk space for dataset

### Installation

1. **Clone the repository and set up the environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download and configure the dataset:**

   - Download the LINEMOD dataset from [this link](https://drive.google.com/drive/u/0/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)
   - Extract the dataset to a location of your choice
   - Update `densefusion/config.py` with your dataset paths:

```python
self.LINEMOD_ROOT = "/path/to/Linemod_preprocessed_yolo_2"
self.PLY_MODELS_DIR = "/path/to/pose_models/models"
self.DIAMETER_INFO_PATH = "/path/to/pose_models/models_info.yml"
self.MODELS_SAVE_DIR = "/path/to/save/models"
```

### Running the Pipeline

```bash
# Train and evaluate (default)
python run_pipeline.py

# Train only
python run_pipeline.py --train

# Evaluate only
python run_pipeline.py --eval

# View help and all available options
python run_pipeline.py --help
```

**Note:** You can use pre-trained YOLOv11 weights from `YOLOv11_finetuning/weights/best.pt` to skip the detection phase.

---

## Project Structure

### Top-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and documentation |
| `requirements.txt` | Python package dependencies |
| `run_pipeline.py` | Main entry point with CLI for training/evaluation |
| `report` | Full technical report with results and analysis |

### Core Package: `densefusion/`

| Module | Description |
|--------|-------------|
| `config.py` | Configuration parameters, paths, and hyperparameters |
| `dataset.py` | `DenseFusionDataset` class with data loading and augmentation |
| `models.py` | DenseFusion model wrappers and architecture definitions |
| `loss.py` | Loss functions (ADD loss, confidence BCE, regularization) |
| `utils.py` | Utility functions (geometry, metrics, I/O) |
| `train.py` | Training loop with checkpointing and mixed precision |
| `eval.py` | Evaluation loop and metric aggregation |
| `visualize.py` | Visualization helpers for results |
| `segmentation.py` | Mask R-CNN wrapper for segmentation refinement |

### Core Models: `densefusion/core_models/`

| File | Purpose |
|------|---------|
| `rgb_encoder.py` | CNN-based RGB feature encoder |
| `pointnet.py` | PointNet architecture for point cloud encoding |
| `pose_estimator.py` | MLP head for pose and confidence regression |
| `transformer_fusion.py` | Custom Transformer-based fusion module (MHSA implementation) |

### Additional Directories

- **`YOLOv11_finetuning/`**: Pre-trained YOLOv11 weights and fine-tuning artifacts
- **`Jupyter_notebook_folder/`**: Original exploratory notebook (`GROUP_30_DENSEFUSION_TRS.ipynb`)
- **`report/`**:  Documentation and technical report


---

## Technical Contributions

### Custom Implementation Highlights

1. **Multi-Head Self-Attention (MHSA)**: Implemented from scratch for transparency and reviewability, rather than using black-box PyTorch modules
2. **TransformerFuser Module**: Compact transformer encoder using:
   - Learnable CLS token for feature fusion
   - Positional embeddings for geometric awareness
   - Multi-layer encoder for progressive fusion

3. **Modular Architecture**: Clean separation of concerns enabling:
   - Easy comparison between MLP and Transformer fusion strategies
   - Straightforward extension with new components
   - Clear data flow for reproducibility

### Core Architecture

- **RGB Encoder**: CNN producing global RGB feature vectors
- **PointNet Encoder**: MLP-based processing of point clouds from depth data
- **Fusion Module**: Either MLP concatenation (baseline) or Transformer-based (proposed)
- **Pose Head**: MLP predicting 6D pose (translation + quaternion) and confidence score

---

## Evaluation

### Metrics Implementation

- **ADD (Average Distance)**: Implemented in `densefusion/utils.py` and `densefusion/loss.py`
- **ADD-S (Symmetric ADD)**: For symmetric objects (driller, eggbox)
- **Success Rates**: Computed at multiple thresholds (2cm, 5cm, 10cm and relative %)

Run evaluation with:
```bash
python run_pipeline.py --eval
```

### Integration with YOLO

The pipeline supports YOLOv11-based detection:
- Fine-tuned weights available in `YOLOv11_finetuning/weights/best.pt`
- Automatic loading during evaluation
- Can be skipped in favor of ground-truth bounding boxes for development

---

## Authors

- **Ismail Abouelseoud**
- **Valeria Intini**
- **Gabriele Parisini**
- **Edoardo Ciorra**

---

## References & Documentation

For detailed experimental settings, numerical results, and comprehensive analysis, refer to the full technical report:

📄 **Full Report**: `report_final.pdf`

This report includes:
- Detailed methodology and implementation choices
- Comprehensive experimental results with ablation studies
- Comparative analysis of MLP vs. Transformer fusion
- Discussion of limitations and future work 



