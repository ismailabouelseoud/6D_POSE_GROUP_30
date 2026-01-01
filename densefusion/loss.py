import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import config


class DenseFusionLoss(nn.Module):
    def __init__(self, object_models, sym_list, add_weight=1.0, conf_weight=0.1):
        super(DenseFusionLoss, self).__init__()
        self.object_models = object_models
        self.add_weight = add_weight
        self.conf_weight = conf_weight
        self.conf_loss = nn.BCEWithLogitsLoss()
        self.sym_list = sym_list
        self.reset_stats()

    def reset_stats(self):
        self.stats = {'total_samples': 0, 'add_computed': 0, 'pose_fallback': 0, 'avg_add_error': 0.0}

    def quaternion_to_matrix(self, q):
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q.unbind(-1)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        matrix = torch.stack([
            1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
            2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
            2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)
        ], dim=-1).view(q.size(0), 3, 3)
        return matrix.squeeze(0) if matrix.size(0) == 1 else matrix

    def compute_add_loss(self, pred_pose, gt_pose, class_id):
        if class_id not in self.object_models:
            return torch.tensor(0.0, device=pred_pose.device)
        vertices = torch.tensor(self.object_models[class_id]['vertices_raw'], device=pred_pose.device, dtype=torch.float32)
        if vertices.shape[0] < 10:
            return torch.tensor(0.0, device=pred_pose.device)
        t_pred = pred_pose[:3]
        q_pred = pred_pose[3:]
        t_gt = gt_pose[:3]
        q_gt = gt_pose[3:]
        R_pred = self.quaternion_to_matrix(q_pred)
        R_gt = self.quaternion_to_matrix(q_gt)
        pred_pts = vertices @ R_pred.T + t_pred
        gt_pts = vertices @ R_gt.T + t_gt
        dists = torch.norm(pred_pts - gt_pts, p=2, dim=1)
        add_loss = dists.mean()
        if add_loss == float('inf') or torch.isnan(add_loss.detach()):
            print("ERROR: CAN'T COMPUTE ADD LOSS")
        self.stats['add_computed'] += 1
        self.stats['avg_add_error'] += add_loss.item()
        return add_loss

    def compute_add_s_loss(self, pred_pose, gt_pose, class_id):
        if class_id not in self.object_models:
            return torch.tensor(0.0, device=pred_pose.device)
        vertices = torch.tensor(self.object_models[class_id]['vertices_raw'], device=pred_pose.device, dtype=torch.float32)
        if vertices.shape[0] < 10:
            return torch.tensor(0.0, device=pred_pose.device)
        t_pred = pred_pose[:3]
        q_pred = pred_pose[3:]
        t_gt = gt_pose[:3]
        q_gt = gt_pose[3:]
        R_pred = self.quaternion_to_matrix(q_pred)
        R_gt = self.quaternion_to_matrix(q_gt)
        pred_pts = vertices @ R_pred.T + t_pred
        gt_pts = vertices @ R_gt.T + t_gt
        dists = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
        add_loss = dists.min(dim=1)[0].mean()
        if add_loss == float('inf') or torch.isnan(add_loss.detach()):
            print("ERROR: CAN'T COMPUTE ADD-S LOSS")
        self.stats['add_computed'] += 1
        self.stats['avg_add_error'] += add_loss.item()
        return add_loss

    def forward(self, pred_poses, gt_poses, pred_confidences, class_ids):
        batch_size = pred_poses.size(0)
        self.stats['total_samples'] += batch_size
        total_add_loss = 0.0
        valid_samples = 0
        for i in range(batch_size):
            class_id = class_ids[i].item()
            if class_id in self.sym_list:
                add_loss_val = self.compute_add_s_loss(pred_poses[i], gt_poses[i], class_id)
            else:
                add_loss_val = self.compute_add_loss(pred_poses[i], gt_poses[i], class_id)
            total_add_loss += add_loss_val
            valid_samples += 1
        avg_add_loss = total_add_loss / batch_size
        conf_targets = torch.ones_like(pred_confidences)
        conf_loss = self.conf_loss(pred_confidences, conf_targets)
        pose_reg_loss = torch.tensor(0.0, device=pred_poses.device)
        for i in range(batch_size):
            trans_magnitude = torch.norm(pred_poses[i, :3])
            if trans_magnitude > 2.0:
                pose_reg_loss += (trans_magnitude - 2.0) ** 2
        pose_reg_loss = pose_reg_loss / batch_size
        total_loss = (self.add_weight * avg_add_loss + self.conf_weight * conf_loss + 0.1 * pose_reg_loss)
        loss_dict = {'total_loss': total_loss.item(), 'add_loss': avg_add_loss.item() if hasattr(avg_add_loss, 'item') else float(avg_add_loss), 'conf_loss': conf_loss.item(), 'valid_samples': valid_samples}
        return total_loss, loss_dict

    def get_stats(self):
        if self.stats['total_samples'] > 0:
            return {'total_samples': self.stats['total_samples'], 'add_computed_ratio': self.stats['add_computed'] / self.stats['total_samples'], 'pose_fallback_ratio': self.stats['pose_fallback'] / self.stats['total_samples'], 'avg_add_error': self.stats['avg_add_error'] / max(self.stats['add_computed'], 1)}
        return self.stats
