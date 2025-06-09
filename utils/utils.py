import os
import torch
import numpy as np
import yaml
import json
import gc
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

class Config:
    """Configuration class for DenseFusion"""
    def __init__(self):
        # --- PATHS (IMPORTANT: UPDATE THESE FOR YOUR SETUP) ---
        self.LINEMOD_ROOT = "/content/datasets/linemod/Linemod_preprocessed_yolo_2"
        self.PLY_MODELS_DIR = "/content/datasets/linemod/Linemod_preprocessed_yolo_2/pose_models/models"
        self.DIAMETER_INFO_PATH = "/content/datasets/linemod/Linemod_preprocessed_yolo_2/pose_models/models_info.yml"
        self.YOLO_MODEL_PATH = "/content/YOLOv11_finetuning/weights/best.pt" # From your finetuning
        self.MODELS_SAVE_DIR = "trained_models" # Saves to root of repo
        self.CHECKPOINTS_DIR = "checkpoints" # Saves to root of repo

        # --- TRAINING HYPERPARAMETERS ---
        self.BATCH_SIZE = 12
        self.NUM_EPOCHS = 30
        self.LEARNING_RATE = 1e-4
        self.NUM_POINTS = 1000
        self.PATCH_SIZE = 224 # Smaller patch size often works well

        # --- MODEL CONFIGURATION ---
        self.USE_SEGMENTATION = True
        self.USE_MIXED_PRECISION = torch.cuda.is_available()
        
        # --- TRANSFORMER FUSION OPTIONS ---
        self.USE_TRANSFORMER_FUSION = True
        self.TRANSFORMER_HEADS = 8
        self.TRANSFORMER_LAYERS = 6
        self.TRANSFORMER_DIM = 64
        self.TRANSFORMER_FFN_DIM = 1024
        self.TRANSFORMER_DROPOUT = 0.1

        # --- EVALUATION ---
        self.MAX_EVAL_SAMPLES = 500

        # --- CONSTANTS ---
        self.MODEL_SCALE_MM_TO_M = 0.001
        self.DEPTH_SCALE_MM_TO_M = 1000.0
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]], dtype=np.float32)

    def setup_environment(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def verify_paths(self):
        """Verify all required paths exist"""
        # Note: We don't check for save/checkpoint dirs as they are created by the script
        paths = { 'LINEMOD dataset': self.LINEMOD_ROOT, 'YOLO model': self.YOLO_MODEL_PATH,
                  'PLY models': self.PLY_MODELS_DIR, 'Diameter info': self.DIAMETER_INFO_PATH }
        all_good = True
        for name, path in paths.items():
            if not os.path.exists(path):
                print(f"✗ {name} NOT FOUND: {path}")
                all_good = False
        return all_good

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_dataset_config(linemod_root):
    """Load dataset configuration from data.yaml"""
    config_path = os.path.join(linemod_root, 'data.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_diameters(diameter_yml_path, scale):
    """Load object model diameters from YAML file"""
    with open(diameter_yml_path, 'r') as f:
        diameter_data = yaml.safe_load(f)
    model_diameters = {}
    for class_id, info in diameter_data.items():
        # Class IDs in YAML are 1-based, convert to 0-based
        internal_class_id = int(class_id) - 1 
        model_diameters[internal_class_id] = float(info['diameter']) * scale
    return model_diameters

def load_yolo_model(model_path):
    """Load and validate YOLO model"""
    try:
        yolo_model = YOLO(model_path)
        print(f"✓ YOLO model loaded successfully from {model_path}")
        return yolo_model
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return None

def load_trained_model(config):
    """Loads a trained DenseFusion model for evaluation."""
    model_path = os.path.join(config.MODELS_SAVE_DIR, 'densefusion_best.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    try:
        dataset_config = load_dataset_config(config.LINEMOD_ROOT)
        num_classes = len(dataset_config.get('names', []))
        
        # Must import here to avoid circular dependency
        from models.densefusion import DenseFusionNetwork
        model = DenseFusionNetwork(num_objects=num_classes, use_transformer=config.USE_TRANSFORMER_FUSION, config=config)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE).eval()
        print(f"✓ Trained DenseFusion model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load trained model: {e}")
        return None

def decompose_pose_numpy(pose_numpy):
    """Decompose 7D pose [t, q_wxyz] into rotation matrix and translation vector."""
    t = pose_numpy[:3]
    q_wxyz = pose_numpy[3:]
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    R_mat = Rotation.from_quat(q_xyzw).as_matrix()
    return R_mat, t

def compute_add_metric(pred_pose, gt_pose, model_vertices, config):
    """Compute ADD metric."""
    if model_vertices is None or model_vertices.shape[0] == 0: return float('inf')
    R_pred, t_pred = decompose_pose_numpy(pred_pose)
    R_gt, t_gt = decompose_pose_numpy(gt_pose)
    
    pred_transformed = (R_pred @ model_vertices.T).T + t_pred
    gt_transformed = (R_gt @ model_vertices.T).T + t_gt
    
    distances = np.linalg.norm(pred_transformed - gt_transformed, axis=1)
    return np.mean(distances)

def compute_add_metrics_with_thresholds(pred_pose, gt_pose, model_vertices, config, diameter=None):
    """Compute ADD metric with various success thresholds."""
    add_value = compute_add_metric(pred_pose, gt_pose, model_vertices, config)
    results = {
        "add_value": add_value,
        "add_success_2cm": add_value < 0.02,
        "add_success_5cm": add_value < 0.05,
        "add_success_10cm": add_value < 0.10,
    }
    if diameter is not None and diameter > 0:
        results.update({
            "add_success_5p": add_value < (0.05 * diameter),
            "add_success_10p": add_value < (0.10 * diameter),
            "add_success_20p": add_value < (0.20 * diameter),
        })
    return results

def convert_yolo_bbox_to_pixel(bbox_normalized, width, height):
    """Convert YOLO normalized bbox to pixel coordinates [x1, y1, x2, y2]."""
    xc_n, yc_n, w_n, h_n = bbox_normalized
    x1 = int((xc_n - w_n / 2) * width)
    y1 = int((yc_n - h_n / 2) * height)
    x2 = int((xc_n + w_n / 2) * width)
    y2 = int((yc_n + h_n / 2) * height)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

def create_directories(config):
    """Create necessary directories for saving models and checkpoints."""
    os.makedirs(config.MODELS_SAVE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

def save_model_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, is_best, config):
    """Save model checkpoint."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    if is_best:
        simple_path = os.path.join(config.MODELS_SAVE_DIR, 'densefusion_best.pth')
        torch.save(model.state_dict(), simple_path)
        print(f"✓ Best model state dict saved to: {simple_path}")
    
    epoch_path = os.path.join(config.CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint_data, epoch_path)
    
def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    """Loads model state from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch