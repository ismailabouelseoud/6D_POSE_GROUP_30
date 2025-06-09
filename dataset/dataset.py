# FILE: dataset/dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
from scipy.spatial.transform import Rotation as R
import trimesh

# Local imports
from utils.utils import convert_yolo_bbox_to_pixel

class DenseFusionDataset(Dataset):
    """Dataset for DenseFusion training and evaluation"""
    def __init__(self, data_config, split, config, use_augmentation=False, use_segmentation=False):
        self.data_config = data_config
        self.split = split
        self.config = config
        self.use_augmentation = use_augmentation and split == 'train'
        self.use_segmentation = use_segmentation
        
        self.rgb_dir = os.path.join(self.config.LINEMOD_ROOT, 'images', split)
        self.depth_dir = os.path.join(self.config.LINEMOD_ROOT, 'depth', split)
        
        self.rgb_paths = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        print(f"Found {len(self.rgb_paths)} images for '{split}' split.")

        self.object_models = self._load_raw_models(data_config.get('names', []))
        
        # This would be the place to initialize the segmentation module if needed per-worker
        # For simplicity, we'll pass it in during evaluation.
        self.segmentation_module = None

    def _load_raw_models(self, object_names):
        models = {}
        for obj_idx, obj_name in enumerate(object_names):
            ply_path = os.path.join(self.config.PLY_MODELS_DIR, f"obj_{obj_idx+1:02d}.ply")
            if os.path.exists(ply_path):
                mesh = trimesh.load_mesh(ply_path, process=False)
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                if vertices.shape[0] > self.config.NUM_POINTS:
                    indices = np.random.choice(vertices.shape[0], self.config.NUM_POINTS, replace=False)
                    vertices = vertices[indices]
                models[obj_idx] = {'name': obj_name, 'vertices_raw': vertices * self.config.MODEL_SCALE_MM_TO_M}
        print(f"✓ Loaded models for {len(models)} objects")
        return models

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        try:
            gt_class_id, gt_pose_7d, bbox_norm = self.load_ground_truth(rgb_path)
            
            rgb_image = cv2.imread(rgb_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            depth_image = None
            depth_path = self.get_depth_path(rgb_path)
            if depth_path:
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_image is not None:
                    depth_image = depth_image.astype(np.float32) / self.config.DEPTH_SCALE_MM_TO_M

            # Segmentation is handled in eval, not during training data loading for speed
            object_mask = None

            rgb_patch, depth_patch, bbox_pixel = self.extract_patches_with_segmentation(
                rgb_image, depth_image, bbox_norm, object_mask
            )

            points_3d = self.depth_to_pointcloud(depth_patch, bbox_pixel)
            
            rgb_tensor = torch.from_numpy(rgb_patch.transpose(2, 0, 1)).float() / 255.0
            points_tensor = torch.from_numpy(points_3d).float()
            gt_pose_tensor = torch.from_numpy(gt_pose_7d).float()
            class_id_tensor = torch.tensor(gt_class_id, dtype=torch.long)

            return {
                'rgb': rgb_tensor, 'points': points_tensor, 
                'class_id': class_id_tensor, 'gt_pose': gt_pose_tensor
            }
        except Exception as e:
            print(f"Error loading sample {idx} ({os.path.basename(rgb_path)}): {e}")
            # Return a dummy sample
            return {
                'rgb': torch.zeros((3, self.config.PATCH_SIZE, self.config.PATCH_SIZE)),
                'points': torch.zeros((self.config.NUM_POINTS, 3)),
                'class_id': torch.tensor(0, dtype=torch.long),
                'gt_pose': torch.tensor([0.0]*7, dtype=torch.float32)
            }


    def load_ground_truth(self, rgb_path):
        """Loads GT from the corresponding label file."""
        label_path = rgb_path.replace('/images/', '/labels/').replace('.png', '.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # First line is bbox, second is rotation, third is translation
        bbox_line = lines[0].strip().split()
        rot_line = lines[1].strip().split()
        trans_line = lines[2].strip().split()

        class_id = int(bbox_line[0]) - 1
        bbox_normalized = [float(x) for x in bbox_line[1:5]]
        
        gt_r_flat = np.array([float(x) for x in rot_line[1:]], dtype=np.float32)
        gt_t = np.array([float(x) for x in trans_line[1:]], dtype=np.float32)

        rotation_matrix = gt_r_flat.reshape((3, 3))
        rot = R.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

        gt_pose_7d = np.concatenate([gt_t / 1000.0, quat_wxyz]) # Convert translation to meters

        return class_id, gt_pose_7d, bbox_normalized
        
    def get_depth_path(self, rgb_path):
        """Gets corresponding depth image path."""
        depth_filename = os.path.basename(rgb_path)
        return os.path.join(self.depth_dir, depth_filename)

    def extract_patches_with_segmentation(self, rgb, depth, bbox_norm, mask=None):
        """Extracts RGB and depth patches."""
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = convert_yolo_bbox_to_pixel(bbox_norm, w, h)
        
        rgb_patch = rgb[y1:y2, x1:x2].copy()
        if mask is not None:
            mask_patch = mask[y1:y2, x1:x2]
            if rgb_patch.shape[:2] == mask_patch.shape:
                rgb_patch[mask_patch == 0] = 0

        rgb_patch = cv2.resize(rgb_patch, (self.config.PATCH_SIZE, self.config.PATCH_SIZE))
        
        depth_patch = None
        if depth is not None:
            depth_patch = depth[y1:y2, x1:x2].copy()
            if mask is not None:
                mask_patch = mask[y1:y2, x1:x2]
                if depth_patch.shape == mask_patch.shape:
                    depth_patch[mask_patch == 0] = 0
            depth_patch = cv2.resize(depth_patch, (self.config.PATCH_SIZE, self.config.PATCH_SIZE), interpolation=cv2.INTER_NEAREST)

        return rgb_patch, depth_patch, [x1, y1, x2, y2]

    def depth_to_pointcloud(self, depth_patch, bbox_pixel):
        """Converts depth patch to a point cloud."""
        if depth_patch is None:
            return np.random.randn(self.config.NUM_POINTS, 3).astype(np.float32) * 0.01

        h, w = depth_patch.shape
        fx, fy, cx, cy = self.config.K[0, 0], self.config.K[1, 1], self.config.K[0, 2], self.config.K[1, 2]
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x1, y1, x2, y2 = bbox_pixel
        scale_x = (x2 - x1) / w if w > 0 else 1
        scale_y = (y2 - y1) / h if h > 0 else 1
        x_coords_orig = x_coords * scale_x + x1
        y_coords_orig = y_coords * scale_y + y1
        
        z_flat = depth_patch.flatten()
        valid_mask = (z_flat > 0)
        
        if not np.any(valid_mask):
            return np.random.randn(self.config.NUM_POINTS, 3).astype(np.float32) * 0.01
        
        x_valid = x_coords_orig.flatten()[valid_mask]
        y_valid = y_coords_orig.flatten()[valid_mask]
        z_valid = z_flat[valid_mask]
        
        points_x = (x_valid - cx) * z_valid / fx
        points_y = (y_valid - cy) * z_valid / fy
        points_3d = np.column_stack((points_x, points_y, z_valid)).astype(np.float32)
        
        if len(points_3d) > self.config.NUM_POINTS:
            indices = np.random.choice(len(points_3d), self.config.NUM_POINTS, replace=False)
            points_3d = points_3d[indices]
        elif len(points_3d) < self.config.NUM_POINTS:
            pad_indices = np.random.choice(len(points_3d), self.config.NUM_POINTS - len(points_3d), replace=True)
            points_3d = np.vstack([points_3d, points_3d[pad_indices]])
            
        return points_3d
