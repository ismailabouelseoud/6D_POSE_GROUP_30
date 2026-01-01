import os
import yaml
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from .config import config


def load_yaml_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def convert_bbox_to_yolo(bbox_linemod, image_width, image_height):
    if not isinstance(bbox_linemod, (list, np.ndarray)) or len(bbox_linemod) != 4:
        return None
    x_min, y_min, width_px, height_px = bbox_linemod
    center_x = (x_min + width_px / 2.0) / image_width
    center_y = (y_min + height_px / 2.0) / image_height
    width_norm = width_px / image_width
    height_norm = height_px / image_height
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    if width_norm <= 0 or height_norm <= 0:
        return None
    return [center_x, center_y, width_norm, height_norm]


def load_depth_scale_factor(dataset_root, folder_id):
    camera_path = os.path.join(dataset_root, 'data', folder_id, 'camera.yml')
    if os.path.exists(camera_path):
        try:
            camera_data = load_yaml_file(camera_path)
            if camera_data:
                return camera_data.get('depth_scale', 1000.0)
        except Exception:
            pass
    return 1000.0


def decompose_pose_numpy(pose_numpy):
    t = pose_numpy[:3]
    q_wxyz = pose_numpy[3:]
    norm_q = np.linalg.norm(q_wxyz)
    if norm_q < 1e-6:
        return np.identity(3), t
    q_wxyz = q_wxyz / norm_q
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    return rot, t


def compute_add_metric(pred_pose_numpy, gt_pose_numpy, model_vertices):
    if model_vertices is None or model_vertices.shape[0] == 0:
        return float('inf')
    try:
        R_pred, t_pred = decompose_pose_numpy(pred_pose_numpy)
        R_gt, t_gt = decompose_pose_numpy(gt_pose_numpy)
        model_points_meters = model_vertices * config.MODEL_SCALE_MM_TO_M
        pred_transformed = (R_pred @ model_points_meters.T).T + t_pred
        gt_transformed = (R_gt @ model_points_meters.T).T + t_gt
        distances = np.linalg.norm(pred_transformed - gt_transformed, axis=1)
        return np.mean(distances)
    except Exception as e:
        print(f"ADD computation error: {e}")
        return float('inf')


def compute_add_s_metric(pred_pose_numpy, gt_pose_numpy, model_vertices):
    if model_vertices is None or model_vertices.shape[0] == 0:
        return float('inf')
    try:
        R_pred, t_pred = decompose_pose_numpy(pred_pose_numpy)
        R_gt, t_gt = decompose_pose_numpy(gt_pose_numpy)
        model_points_meters = model_vertices * config.MODEL_SCALE_MM_TO_M
        pred_transformed = (R_pred @ model_points_meters.T).T + t_pred
        gt_transformed = (R_gt @ model_points_meters.T).T + t_gt
        gt_kdtree = cKDTree(gt_transformed)
        distances, _ = gt_kdtree.query(pred_transformed, k=1)
        return np.mean(distances)
    except Exception as e:
        print(f"ADD-S computation error: {e}")
        return float('inf')


def compute_add_metrics_with_thresholds(pred_pose, gt_pose, class_id, sym_list, model_vertices, diameter=None):
    if class_id in sym_list:
        add_value = compute_add_s_metric(pred_pose, gt_pose, model_vertices)
    else:
        add_value = compute_add_metric(pred_pose, gt_pose, model_vertices)
    results = {
        "add_value": add_value,
        "add_success_2cm": add_value < 0.02,
        "add_success_5cm": add_value < 0.05,
        "add_success_10cm": add_value < 0.10,
    }
    if diameter is not None and diameter > 0:
        results.update({
            "diameter": diameter,
            "add_success_5p": add_value < (0.05 * diameter),
            "add_success_10p": add_value < (0.10 * diameter),
            "add_success_20p": add_value < (0.20 * diameter),
        })
    return results


def compute_rotation_difference_degrees(pred_pose, gt_pose):
    try:
        pred_quat = pred_pose[3:] / np.linalg.norm(pred_pose[3:])
        gt_quat = gt_pose[3:] / np.linalg.norm(gt_pose[3:])
        pred_quat_scipy = [pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]]
        gt_quat_scipy = [gt_quat[1], gt_quat[2], gt_quat[3], gt_quat[0]]
        pred_rot = R.from_quat(pred_quat_scipy)
        gt_rot = R.from_quat(gt_quat_scipy)
        relative_rot = pred_rot * gt_rot.inv()
        overall_angle_deg = np.degrees(relative_rot.magnitude())
        pred_euler = pred_rot.as_euler('xyz', degrees=True)
        gt_euler = gt_rot.as_euler('xyz', degrees=True)
        diff_x = min(abs(pred_euler[0] - gt_euler[0]), 360 - abs(pred_euler[0] - gt_euler[0]))
        diff_y = min(abs(pred_euler[1] - gt_euler[1]), 360 - abs(pred_euler[1] - gt_euler[1]))
        diff_z = min(abs(pred_euler[2] - gt_euler[2]), 360 - abs(pred_euler[2] - gt_euler[2]))
        return {
            'overall': overall_angle_deg,
            'x_axis': diff_x,
            'y_axis': diff_y,
            'z_axis': diff_z,
            'pred_euler': pred_euler,
            'gt_euler': gt_euler
        }
    except Exception as e:
        return {'overall': float('inf'), 'x_axis': 0, 'y_axis': 0, 'z_axis': 0}


def convert_yolo_bbox_to_pixel(bbox_normalized, image_width, image_height):
    xc_n, yc_n, w_n, h_n = bbox_normalized
    xc_px = xc_n * image_width
    yc_px = yc_n * image_height
    w_px = w_n * image_width
    h_px = h_n * image_height
    x1 = max(0, int(xc_px - w_px / 2))
    y1 = max(0, int(yc_px - h_px / 2))
    x2 = min(image_width, int(xc_px + w_px / 2))
    y2 = min(image_height, int(yc_px + h_px / 2))
    return [x1, y1, x2, y2]
