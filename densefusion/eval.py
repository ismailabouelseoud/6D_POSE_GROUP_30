import os
import json
import numpy as np
import torch
from tqdm import tqdm
from .config import config
from .dataset import DenseFusionDataset
from .utils import load_yaml_file, compute_add_metrics_with_thresholds


def detect_and_estimate_pose(yolo_model, pose_model, dataset, rgb_path, depth_path=None):
    try:
        import cv2
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            return None
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = yolo_model(rgb_image, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        result = results[0]
        box = result.boxes[0]
        conf = float(box.conf)
        yolo_class_id = int(box.cls)
        class_id = yolo_class_id - 1
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        h, w = rgb_image.shape[:2]
        xc_n = (x1 + x2) / (2 * w)
        yc_n = (y1 + y2) / (2 * h)
        w_n = (x2 - x1) / w
        h_n = (y2 - y1) / h
        bbox_normalized = [xc_n, yc_n, w_n, h_n]
        object_mask = None
        if hasattr(dataset, 'segmentation_module') and dataset.segmentation_module is not None:
            try:
                object_mask, _ = dataset.segmentation_module.refine_detection(rgb_image, [x1, y1, x2, y2], class_id)
            except Exception:
                object_mask = None
        depth_image = None
        if depth_path and os.path.exists(depth_path):
            import cv2
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is not None:
                depth_image = depth_image.astype(np.float32) / config.DEPTH_SCALE_MM_TO_M
        rgb_patch, depth_patch, bbox_pixel = dataset.extract_patches_with_segmentation(rgb_image, depth_image, bbox_normalized, object_mask)
        points_3d = dataset.depth_to_pointcloud(depth_patch, bbox_pixel)
        rgb_tensor = torch.from_numpy(rgb_patch.transpose(2, 0, 1)).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(config.DEVICE)
        points_tensor = torch.from_numpy(points_3d).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            pred_pose, pred_conf = pose_model(rgb_tensor, points_tensor)
        pred_pose_np = pred_pose.cpu().numpy().flatten()
        confidence = torch.sigmoid(pred_conf).cpu().numpy().item()
        return {'bbox': [x1, y1, x2, y2], 'class_id': class_id, 'pose': pred_pose_np, 'confidence': confidence, 'yolo_confidence': conf, 'mask': object_mask}
    except Exception as e:
        print(f"Error in pose estimation: {e}")
        return None


def evaluate_model_comprehensive(yolo_model, pose_model, test_dataset, model_diameters=None):
    num_samples = min(config.NUM_POINTS, len(test_dataset))
    pose_model.eval()
    metrics = {'add_values': [], 'add_2cm': [], 'add_5cm': [], 'add_10cm': [], 'yolo_confidence': [], 'pose_confidence': [], 'class_ids': [], 'success_by_class': {}, 'detection_rate': 0, 'pose_estimates': []}
    if model_diameters is not None:
        metrics.update({'add_5p': [], 'add_10p': [], 'add_20p': []})
    for i in tqdm(range(num_samples), desc="Evaluating"):
        try:
            sample = test_dataset[i]
            rgb_path = test_dataset.rgb_paths[i]
            gt_pose = sample['gt_pose'].cpu().numpy()
            class_id = sample['class_id'].item()
            depth_path = test_dataset.get_depth_path(rgb_path)
            result = detect_and_estimate_pose(yolo_model, pose_model, test_dataset, rgb_path, depth_path)
            if result is None:
                metrics['add_values'].append(float('inf'))
                metrics['add_2cm'].append(0)
                metrics['add_5cm'].append(0)
                metrics['add_10cm'].append(0)
                metrics['yolo_confidence'].append(0)
                metrics['pose_confidence'].append(0)
                metrics['class_ids'].append(class_id)
                if model_diameters is not None:
                    metrics['add_5p'].append(0); metrics['add_10p'].append(0); metrics['add_20p'].append(0)
                continue
            pred_pose = result['pose']
            yolo_conf = result['yolo_confidence']
            pose_conf = result['confidence']
            if class_id in test_dataset.object_models:
                model_vertices = test_dataset.object_models[class_id]['vertices_raw']
                if model_vertices.shape[0] > 10:
                    diameter = model_diameters.get(class_id, None) if model_diameters else None
                    add_metrics = compute_add_metrics_with_thresholds(pred_pose, gt_pose, class_id, config.SYMMETRIC_LIST, model_vertices, diameter)
                    metrics['add_values'].append(add_metrics['add_value'])
                    metrics['add_2cm'].append(int(add_metrics['add_success_2cm']))
                    metrics['add_5cm'].append(int(add_metrics['add_success_5cm']))
                    metrics['add_10cm'].append(int(add_metrics['add_success_10cm']))
                    if model_diameters is not None and diameter is not None:
                        metrics['add_5p'].append(int(add_metrics['add_success_5p'])); metrics['add_10p'].append(int(add_metrics['add_success_10p'])); metrics['add_20p'].append(int(add_metrics['add_success_20p']))
                    elif model_diameters is not None:
                        metrics['add_5p'].append(0); metrics['add_10p'].append(0); metrics['add_20p'].append(0)
                else:
                    metrics['add_values'].append(float('inf'))
                    metrics['add_2cm'].append(0); metrics['add_5cm'].append(0); metrics['add_10cm'].append(0)
                    if model_diameters is not None:
                        metrics['add_5p'].append(0); metrics['add_10p'].append(0); metrics['add_20p'].append(0)
            else:
                metrics['add_values'].append(float('inf'))
                metrics['add_2cm'].append(0); metrics['add_5cm'].append(0); metrics['add_10cm'].append(0)
                if model_diameters is not None:
                    metrics['add_5p'].append(0); metrics['add_10p'].append(0); metrics['add_20p'].append(0)
            metrics['yolo_confidence'].append(yolo_conf); metrics['pose_confidence'].append(pose_conf); metrics['class_ids'].append(class_id)
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            continue
    detection_count = sum(1 for v in metrics['add_values'] if v < float('inf'))
    total_samples = len(metrics['add_values'])
    metrics['detection_rate'] = detection_count / total_samples if total_samples > 0 else 0
    valid_add_values = [v for v in metrics['add_values'] if v < float('inf')]
    metrics['mean_add'] = np.mean(valid_add_values) if valid_add_values else float('inf')
    metrics['success_rate_2cm'] = np.mean(metrics['add_2cm']) if metrics['add_2cm'] else 0
    metrics['success_rate_5cm'] = np.mean(metrics['add_5cm']) if metrics['add_5cm'] else 0
    metrics['success_rate_10cm'] = np.mean(metrics['add_10cm']) if metrics['add_10cm'] else 0
    print("Evaluation complete")
    return metrics


def run_complete_evaluation():
    print("Running complete evaluation (helper)")
    yolo_model = None
    dataset_config = load_yaml_file(config.DATA_YAML_PATH)
    if dataset_config is None:
        raise FileNotFoundError(
            f"Dataset config not found at: {config.DATA_YAML_PATH}\n"
            f"Please download the dataset from: https://drive.google.com/drive/u/0/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7\n"
            f"Then update LINEMOD_ROOT in densefusion/config.py to point to your downloaded dataset."
        )
    test_dataset = DenseFusionDataset(dataset_config, split='val')
    pose_model = None
    metrics = evaluate_model_comprehensive(yolo_model, pose_model, test_dataset, {})
    return metrics
