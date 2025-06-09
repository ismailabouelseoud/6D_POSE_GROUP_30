#need to decide where to put this one#

import os
import torch
import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

# Local imports
from utils.utils import Config, load_trained_model, load_dataset_config, decompose_pose_numpy
from dataset.dataset import DenseFusionDataset

def compute_rotation_difference_degrees(pred_pose, gt_pose):
    """Compute rotational difference in degrees between two poses."""
    try:
        _, pred_t = decompose_pose_numpy(pred_pose)
        _, gt_t = decompose_pose_numpy(gt_pose)
        
        pred_rot = Rotation.from_matrix(_)
        gt_rot = Rotation.from_matrix(_)

        relative_rot = pred_rot * gt_rot.inv()
        return np.degrees(relative_rot.magnitude())
    except Exception:
        return float('inf')


def visualize_predictions(pose_model, dataset, sample_idx, config):
    """
    Generates an enhanced, multi-panel visualization for a single sample from the dataset.
    It shows the original image with bounding boxes and a 3D plot for each detected object,
    comparing the ground truth pose with the predicted pose.
    """
    print(f"🎨 Generating visualization for sample index: {sample_idx}")
    pose_model.eval()

    # --- 1. Load Data for the Specific Sample ---
    try:
        sample = dataset[sample_idx]
        rgb_path = dataset.rgb_paths[sample_idx]
        rgb_image_orig = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Failed to load sample {sample_idx}: {e}")
        return

    # --- 2. Prepare Tensors and Get Prediction ---
    rgb_tensor = sample['rgb'].unsqueeze(0).to(config.DEVICE)
    points_tensor = sample['points'].unsqueeze(0).to(config.DEVICE)
    gt_pose = sample['gt_pose'].cpu().numpy()
    class_id = sample['class_id'].item()
    object_name = dataset.object_models[class_id]['name']

    with torch.no_grad():
        pred_pose_tensor, pred_conf = pose_model(rgb_tensor, points_tensor)
    pred_pose = pred_pose_tensor[0].cpu().numpy()

    # --- 3. Compute Metrics for Visualization ---
    model_vertices = dataset.object_models[class_id]['vertices_raw']
    
    R_pred, t_pred = decompose_pose_numpy(pred_pose)
    R_gt, t_gt = decompose_pose_numpy(gt_pose)
    
    pred_points = (R_pred @ model_vertices.T).T + t_pred
    gt_points = (R_gt @ model_vertices.T).T + t_gt
    
    add_error = np.mean(np.linalg.norm(pred_points - gt_points, axis=1))
    rot_error = compute_rotation_difference_degrees(pred_pose, gt_pose)
    
    print(f"Object: {object_name} (Class ID: {class_id})")
    print(f"  -> ADD Error: {add_error:.4f} m")
    print(f"  -> Rotation Error: {rot_error:.2f}°")

    # --- 4. Create Visualization Figure ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'scene'}]],
        subplot_titles=(f'Input Image: {os.path.basename(rgb_path)}', 
                        f'{object_name}: ADD={add_error:.3f}m, Rot={rot_error:.1f}°')
    )

    # --- Add Original Image with GT BBox ---
    h, w = rgb_image_orig.shape[:2]
    _, _, bbox_norm = dataset.load_ground_truth(rgb_path)
    x1, y1, x2, y2 = dataset.convert_yolo_bbox_to_pixel(bbox_norm, w, h)
    cv2.rectangle(rgb_image_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    fig.add_trace(go.Image(z=rgb_image_orig), row=1, col=1)

    # --- Add 3D Point Clouds ---
    # Subsample for performance
    if len(gt_points) > 1000:
        indices = np.random.choice(len(gt_points), 1000, replace=False)
        gt_viz, pred_viz = gt_points[indices], pred_points[indices]
    else:
        gt_viz, pred_viz = gt_points, pred_points

    # Ground Truth (Green)
    fig.add_trace(go.Scatter3d(
        x=gt_viz[:, 0], y=gt_viz[:, 1], z=gt_viz[:, 2],
        mode='markers', marker=dict(size=2, color='green', opacity=0.8),
        name='Ground Truth'
    ), row=1, col=2)

    # Prediction (Red)
    fig.add_trace(go.Scatter3d(
        x=pred_viz[:, 0], y=pred_viz[:, 1], z=pred_viz[:, 2],
        mode='markers', marker=dict(size=2, color='red', opacity=0.8),
        name='Prediction'
    ), row=1, col=2)
    
    # --- 5. Finalize Layout ---
    fig.update_layout(
        title_text=f"DenseFusion Visualization (Sample {sample_idx})",
        height=600,
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data', # Ensures correct aspect ratio
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        showlegend=True
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    
    fig.show()


if __name__ == '__main__':
    # --- SETUP ---
    config = Config()
    if not config.verify_paths():
        print("\n⚠ Please update paths in utils/utils.py and run again.")
    else:
        # Load the trained model
        pose_model = load_trained_model(config)
        
        if pose_model:
            # Load the dataset
            dataset_config = load_dataset_config(config.LINEMOD_ROOT)
            # Use 'val' or 'test' split for visualization
            vis_dataset = DenseFusionDataset(dataset_config, split='val', config=config)

            # --- VISUALIZE A SAMPLE ---
            # Change this index to see different results
            SAMPLE_TO_VISUALIZE = 15
            
            if SAMPLE_TO_VISUALIZE < len(vis_dataset):
                visualize_predictions(pose_model, vis_dataset, SAMPLE_TO_VISUALIZE, config)
            else:
                print(f"❌ Sample index {SAMPLE_TO_VISUALIZE} is out of range for the dataset (size: {len(vis_dataset)}).")

