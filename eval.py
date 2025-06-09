# FILE: eval.py
import os
import torch
import numpy as np
import json
import datetime
from tqdm import tqdm

# Local imports
from models.densefusion import DenseFusionNetwork
from models.segmentation import DenseFusionSegmentationModule
from dataset.dataset import DenseFusionDataset
from utils.utils import (
    load_yolo_model,
    load_dataset_config,
    load_model_diameters,
    load_trained_model,
    Config,
    compute_add_metrics_with_thresholds,
    convert_yolo_bbox_to_pixel
)

def detect_and_estimate_pose(yolo_model, pose_model, dataset, rgb_path, segmentation_module, config, depth_path=None):
    """Complete detection and pose estimation pipeline for a single image."""
    import cv2
    try:
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None: return None
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        results = yolo_model(rgb_image, verbose=False)
        if not results or not results[0].boxes: return None

        result = results[0]
        box = result.boxes[0]
        yolo_class_id = int(box.cls)
        class_id = yolo_class_id - 1
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        h, w = rgb_image.shape[:2]
        bbox_normalized = [(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h]

        object_mask = None
        if segmentation_module:
            object_mask, _ = segmentation_module.refine_detection(rgb_image, [x1, y1, x2, y2], class_id)

        depth_image = None
        if depth_path and os.path.exists(depth_path):
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is not None:
                depth_image = depth_image.astype(np.float32) / config.DEPTH_SCALE_MM_TO_M

        rgb_patch, depth_patch, bbox_pixel = dataset.extract_patches_with_segmentation(
            rgb_image, depth_image, bbox_normalized, object_mask
        )
        points_3d = dataset.depth_to_pointcloud(depth_patch, bbox_pixel)

        rgb_tensor = torch.from_numpy(rgb_patch.transpose(2, 0, 1)).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(config.DEVICE)
        points_tensor = torch.from_numpy(points_3d).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            pred_pose, pred_conf = pose_model(rgb_tensor, points_tensor)

        return {
            'class_id': class_id,
            'pose': pred_pose.cpu().numpy().flatten(),
            'confidence': torch.sigmoid(pred_conf).cpu().numpy().item(),
        }
    except Exception as e:
        print(f"Error during pose estimation for {os.path.basename(rgb_path)}: {e}")
        return None


def evaluate_model_comprehensive(yolo_model, pose_model, test_dataset, segmentation_module, config, model_diameters=None):
    """Comprehensive model evaluation with ADD metrics."""
    print("Starting comprehensive evaluation...")
    num_samples = min(config.MAX_EVAL_SAMPLES, len(test_dataset))
    print(f"Evaluating on {num_samples} samples")

    pose_model.eval()
    metrics = {
        'add_values': [], 'add_2cm': [], 'add_5cm': [], 'add_10cm': [],
        'add_5p': [], 'add_10p': [], 'add_20p': [],
        'detection_count': 0, 'success_by_class': {}
    }

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = test_dataset[i]
        rgb_path = test_dataset.rgb_paths[i]
        gt_pose = sample['gt_pose'].cpu().numpy()
        class_id = sample['class_id'].item()
        depth_path = test_dataset.get_depth_path(rgb_path)

        result = detect_and_estimate_pose(
            yolo_model, pose_model, test_dataset, rgb_path, segmentation_module, config, depth_path
        )

        if result is None:
            continue
        
        metrics['detection_count'] += 1
        pred_pose = result['pose']

        if class_id in test_dataset.object_models:
            model_vertices = test_dataset.object_models[class_id]['vertices_raw']
            diameter = model_diameters.get(class_id)
            add_metrics = compute_add_metrics_with_thresholds(pred_pose, gt_pose, model_vertices, config, diameter)

            metrics['add_values'].append(add_metrics['add_value'])
            metrics['add_2cm'].append(int(add_metrics['add_success_2cm']))
            metrics['add_5cm'].append(int(add_metrics['add_success_5cm']))
            metrics['add_10cm'].append(int(add_metrics['add_success_10cm']))
            if diameter:
                metrics['add_5p'].append(int(add_metrics['add_success_5p']))
                metrics['add_10p'].append(int(add_metrics['add_success_10p']))
                metrics['add_20p'].append(int(add_metrics['add_success_20p']))

    # Calculate overall metrics
    total_samples = len(metrics['add_2cm'])
    metrics['detection_rate'] = metrics['detection_count'] / num_samples if num_samples > 0 else 0
    metrics['mean_add'] = np.mean([v for v in metrics['add_values'] if v < float('inf')])
    metrics['success_rate_2cm'] = np.mean(metrics['add_2cm'])
    metrics['success_rate_5cm'] = np.mean(metrics['add_5cm'])
    metrics['success_rate_10cm'] = np.mean(metrics['add_10cm'])
    metrics['success_rate_5p'] = np.mean(metrics['add_5p'])
    metrics['success_rate_10p'] = np.mean(metrics['add_10p'])
    metrics['success_rate_20p'] = np.mean(metrics['add_20p'])

    return metrics


def run_complete_evaluation(config):
    """Run the complete evaluation pipeline."""
    print("=" * 60)
    print("COMPLETE DENSEFUSION EVALUATION")
    print("=" * 60)

    yolo_model = load_yolo_model(config.YOLO_MODEL_PATH)
    if not yolo_model: return

    dataset_config = load_dataset_config(config.LINEMOD_ROOT)
    model_diameters = load_model_diameters(config.DIAMETER_INFO_PATH)
    
    test_dataset = DenseFusionDataset(dataset_config, split='val', config=config)
    
    pose_model = load_trained_model(config)
    if not pose_model: return
    
    segmentation_module = None
    if config.USE_SEGMENTATION:
        segmentation_module = DenseFusionSegmentationModule()

    metrics = evaluate_model_comprehensive(
        yolo_model, pose_model, test_dataset, segmentation_module, config, model_diameters
    )

    # Print and save results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Samples evaluated: {config.MAX_EVAL_SAMPLES}")
    print(f"  Detection rate: {metrics['detection_rate']:.4f}")
    print(f"  Mean ADD error: {metrics['mean_add']:.4f} m")
    print(f"  Success rate (<2cm): {metrics['success_rate_2cm']:.4f}")
    print(f"  Success rate (<5cm): {metrics['success_rate_5cm']:.4f}")
    print(f"  Success rate (<10cm): {metrics['success_rate_10cm']:.4f}")
    print(f"  Success rate (<5% diameter): {metrics['success_rate_5p']:.4f}")
    print(f"  Success rate (<10% diameter): {metrics['success_rate_10p']:.4f}")
    print(f"  Success rate (<20% diameter): {metrics['success_rate_20p']:.4f}")
    print("="*60)
    
    # Save results to JSON
    results_file = os.path.join(config.MODELS_SAVE_DIR, f'evaluation_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to native python types for json serialization
        serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    print(f"✓ Evaluation results saved to: {results_file}")


if __name__ == '__main__':
    config = Config()
    
    # Set to a smaller number for quick tests, or len(test_dataset) for full eval
    config.MAX_EVAL_SAMPLES = 50 
    
    if not config.verify_paths():
        print("\n⚠ Please update the paths in the utils/utils.py file before proceeding")
    else:
        run_complete_evaluation(config)