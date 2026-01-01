import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from .config import config
from .utils import convert_yolo_bbox_to_pixel
from typing import Optional


class DenseFusionDataset(Dataset):
    def __init__(self, data_config, split='train', num_points=None, patch_size=None, use_segmentation=config.USE_SEGMENTATION, segmentation_module=None):
        self.data_config = data_config
        self.split = split
        self.num_points = num_points or config.NUM_POINTS
        self.patch_size = patch_size or config.PATCH_SIZE
        self.use_segmentation = use_segmentation
        self.segmentation_module = segmentation_module
        self.rgb_dir = data_config[split]
        self.depth_dir = data_config.get(f'depth_{split}')
        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        rgb_extensions = ["*.png", "*.jpg", "*.jpeg"]
        all_rgb_paths = []
        for ext in rgb_extensions:
            all_rgb_paths.extend(glob.glob(os.path.join(self.rgb_dir, ext)))
        self.rgb_paths = sorted(all_rgb_paths)
        print(f"Dataset '{split}': {len(self.rgb_paths)} images")
        self.object_models = {}
        self._load_raw_models(data_config.get('names', []))

    def _load_raw_models(self, object_names):
        self.object_models = {}
        for obj_idx, obj_name in enumerate(object_names):
            vertices = None
            if os.path.exists(config.PLY_MODELS_DIR):
                ply_candidates = [
                    os.path.join(config.PLY_MODELS_DIR, f"obj_{obj_idx+1:02d}.ply"),
                    os.path.join(config.PLY_MODELS_DIR, f"obj_{obj_idx+1}.ply"),
                    os.path.join(config.PLY_MODELS_DIR, f"{obj_name}.ply")
                ]
                for ply_path in ply_candidates:
                    if os.path.exists(ply_path):
                        try:
                            mesh = trimesh.load_mesh(ply_path, process=False)
                            vertices = np.asarray(mesh.vertices, dtype=np.float32)
                            if vertices.size > 0:
                                break
                        except Exception:
                            continue
            if vertices is None or vertices.size == 0:
                vertices = np.array([[-20, -20, -20], [20, -20, -20], [20, 20, -20], [-20, 20, -20], [-20, -20, 20], [20, -20, 20], [20, 20, 20], [-20, 20, 20]], dtype=np.float32)
            if vertices.shape[0] > config.NUM_POINTS:
                indices = np.random.choice(vertices.shape[0], config.NUM_POINTS, replace=False)
                vertices = vertices[indices]
            self.object_models[obj_idx] = {'name': obj_name, 'vertices_raw': vertices * 0.001, 'vertices_gt': None}
        print(f"✓ Loaded models for {len(self.object_models)} objects")

    def get_depth_path(self, rgb_path: str) -> Optional[str]:
        if not self.depth_dir:
            return None
        rgb_filename = os.path.basename(rgb_path)
        depth_path = os.path.join(self.depth_dir, rgb_filename)
        return depth_path if os.path.exists(depth_path) else None

    def extract_patches_with_segmentation(self, rgb_image, depth_image, bbox_norm, mask=None):
        h, w = rgb_image.shape[:2]
        xc_n, yc_n, w_n, h_n = bbox_norm
        xc_px = int(xc_n * w)
        yc_px = int(yc_n * h)
        w_px = int(w_n * w)
        h_px = int(h_n * h)
        x1 = max(0, xc_px - w_px // 2)
        y1 = max(0, yc_px - h_px // 2)
        x2 = min(w, xc_px + w_px // 2)
        y2 = min(h, yc_px + h_px // 2)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, min(w, 100), min(h, 100)
        rgb_patch = rgb_image[y1:y2, x1:x2].copy()
        if mask is not None:
            mask_patch = mask[y1:y2, x1:x2]
            if rgb_patch.shape[:2] == mask_patch.shape:
                rgb_patch[mask_patch == 0] = 0
        if rgb_patch.size == 0:
            rgb_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        else:
            rgb_patch = cv2.resize(rgb_patch, (self.patch_size, self.patch_size))
        depth_patch = None
        if depth_image is not None:
            depth_patch = depth_image[y1:y2, x1:x2].copy()
            if mask is not None:
                mask_patch = mask[y1:y2, x1:x2]
                if depth_patch.shape == mask_patch.shape:
                    depth_patch[mask_patch == 0] = 0
            if depth_patch.size > 0:
                depth_patch = cv2.resize(depth_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            else:
                depth_patch = None
        return rgb_patch, depth_patch, [x1, y1, x2, y2]

    def depth_to_pointcloud(self, depth_patch, bbox_pixel):
        if depth_patch is None:
            return np.random.randn(self.num_points, 3).astype(np.float32) * 0.01
        h, w = depth_patch.shape
        if h == 0 or w == 0:
            return np.random.randn(self.num_points, 3).astype(np.float32) * 0.01
        fx, fy = config.K[0, 0], config.K[1, 1]
        cx, cy = config.K[0, 2], config.K[1, 2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x1, y1, x2, y2 = bbox_pixel
        scale_x = (x2 - x1) / w if w > 0 else 1
        scale_y = (y2 - y1) / h if h > 0 else 1
        x_coords_orig = x_coords * scale_x + x1
        y_coords_orig = y_coords * scale_y + y1
        x_flat = x_coords_orig.flatten()
        y_flat = y_coords_orig.flatten()
        z_flat = depth_patch.flatten()
        valid_mask = (z_flat > 0) & (z_flat < 5.0)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        if len(z_valid) == 0:
            return np.random.randn(self.num_points, 3).astype(np.float32) * 0.01
        points_x = (x_valid - cx) * z_valid / fx
        points_y = (y_valid - cy) * z_valid / fy
        points_z = z_valid
        points_3d = np.column_stack((points_x, points_y, points_z))
        if len(points_3d) > self.num_points:
            indices = np.random.choice(len(points_3d), self.num_points, replace=False)
            points_3d = points_3d[indices]
        elif len(points_3d) < self.num_points:
            if len(points_3d) == 0:
                points_3d = np.random.randn(self.num_points, 3).astype(np.float32) * 0.01
            else:
                num_to_pad = self.num_points - len(points_3d)
                pad_indices = np.random.choice(len(points_3d), num_to_pad, replace=True)
                points_3d = np.vstack([points_3d, points_3d[pad_indices]])
        return points_3d.astype(np.float32)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        try:
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                raise FileNotFoundError(f"Could not load RGB image: {rgb_path}")
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            depth_image = None
            depth_path = self.get_depth_path(rgb_path)
            if depth_path:
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_image is not None:
                    depth_image = depth_image.astype(np.float32) / config.DEPTH_SCALE_MM_TO_M
            # Very small GT loader - expect YOLO label format
            yolo_label_path = rgb_path.replace('/images/', '/labels/').replace('.png', '.txt').replace('.jpg', '.txt')
            gt_class_id = 0
            bbox_normalized = [0.5, 0.5, 0.3, 0.3]
            if os.path.exists(yolo_label_path):
                try:
                    with open(yolo_label_path, 'r') as f:
                        for lin in f.readlines():
                            line = lin.strip().split()
                            if len(line) == 5:
                                yolo_class_id = int(line[0])
                                gt_class_id = max(0, min(yolo_class_id - 1, len(self.data_config.get('names', [])) - 1))
                                bbox_normalized = [float(x) for x in line[1:5]]
                                break
                except Exception:
                    pass
            gt_pose_7d = np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            object_mask = None
            if self.use_segmentation and self.segmentation_module is not None:
                try:
                    bbox_pixel = convert_yolo_bbox_to_pixel(bbox_normalized, rgb_image.shape[1], rgb_image.shape[0])
                    object_mask, _ = self.segmentation_module.refine_detection(rgb_image, bbox_pixel, gt_class_id)
                except Exception:
                    object_mask = None
            rgb_patch, depth_patch, bbox_pixel = self.extract_patches_with_segmentation(rgb_image, depth_image, bbox_normalized, object_mask)
            points_3d = self.depth_to_pointcloud(depth_patch, bbox_pixel)
            rgb_tensor = torch.from_numpy(rgb_patch.transpose(2, 0, 1)).float() / 255.0
            points_tensor = torch.from_numpy(points_3d).float()
            gt_pose_tensor = torch.from_numpy(gt_pose_7d).float()
            class_id_tensor = torch.tensor(gt_class_id, dtype=torch.long)
            return {'rgb': rgb_tensor, 'points': points_tensor, 'class_id': class_id_tensor, 'gt_pose': gt_pose_tensor}
        except Exception as e:
            print(f"Error loading sample: {e}")
            default_rgb = torch.zeros((3, self.patch_size, self.patch_size), dtype=torch.float32)
            default_points = torch.zeros((self.num_points, 3), dtype=torch.float32)
            default_class_id = torch.tensor(0, dtype=torch.long)
            default_pose = torch.tensor([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            return {'rgb': default_rgb, 'points': default_points, 'class_id': default_class_id, 'gt_pose': default_pose}
