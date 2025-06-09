import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseFusionLoss(nn.Module):
    """Complete loss function for DenseFusion training."""

    def __init__(self, object_models, config, add_weight=1.0, conf_weight=0.1):
        super(DenseFusionLoss, self).__init__()
        self.object_models = object_models
        self.config = config
        self.add_weight = add_weight
        self.conf_weight = conf_weight
        self.conf_loss = nn.BCEWithLogitsLoss()
        self.reset_stats()

    def reset_stats(self):
        """Reset statistics for tracking."""
        self.stats = {'total_samples': 0, 'add_computed': 0, 'avg_add_error': 0.0}

    def quaternion_to_matrix(self, q):
        """Batched quaternion [qw, qx, qy, qz] to rotation matrix conversion."""
        w, x, y, z = q.unbind(-1)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return torch.stack([
            1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
            2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
            2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)
        ], dim=-1).view(q.size(0), 3, 3)

    def compute_add_loss(self, pred_pose, gt_pose, class_id):
        """Compute ADD loss for a single sample."""
        if class_id.item() not in self.object_models:
            return self._pose_distance_loss(pred_pose, gt_pose)
            
        vertices = torch.tensor(
            self.object_models[class_id.item()]['vertices_raw'],
            device=pred_pose.device, dtype=torch.float32
        )
        if vertices.shape[0] < 10:
            return self._pose_distance_loss(pred_pose, gt_pose)

        t_pred, q_pred = pred_pose[:3], pred_pose[3:]
        t_gt, q_gt = gt_pose[:3], gt_pose[3:]
        
        R_pred = self.quaternion_to_matrix(q_pred.unsqueeze(0)).squeeze(0)
        R_gt = self.quaternion_to_matrix(q_gt.unsqueeze(0)).squeeze(0)

        pred_pts = vertices @ R_pred.T + t_pred
        gt_pts = vertices @ R_gt.T + t_gt

        dists = torch.norm(pred_pts - gt_pts, p=2, dim=1)
        add_loss = dists.mean()
        
        self.stats['add_computed'] += 1
        self.stats['avg_add_error'] += add_loss.item()
        return add_loss

    def _pose_distance_loss(self, pred_pose, gt_pose):
        """Fallback loss for symmetric objects or when ADD fails."""
        trans_loss = F.l1_loss(pred_pose[:3], gt_pose[:3])
        dot_product = torch.abs(torch.sum(pred_pose[3:] * gt_pose[3:]))
        quat_loss = 1.0 - dot_product
        return trans_loss + quat_loss

    def forward(self, pred_poses, gt_poses, pred_confidences, class_ids):
        """Compute total loss for a batch."""
        batch_size = pred_poses.size(0)
        self.stats['total_samples'] += batch_size
        
        add_losses = [self.compute_add_loss(pred_poses[i], gt_poses[i], class_ids[i]) for i in range(batch_size)]
        avg_add_loss = torch.stack(add_losses).mean()
        
        conf_targets = torch.ones_like(pred_confidences)
        conf_loss = self.conf_loss(pred_confidences, conf_targets)
        
        total_loss = self.add_weight * avg_add_loss + self.conf_weight * conf_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'add_loss': avg_add_loss.item(),
            'conf_loss': conf_loss.item()
        }
        return total_loss, loss_dict
