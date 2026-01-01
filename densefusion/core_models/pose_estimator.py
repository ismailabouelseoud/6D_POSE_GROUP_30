import torch
import torch.nn as nn
import torch.nn.functional as F
from .rgb_encoder import RGBEncoder
from .pointnet import PointNetEncoder
from .transformer_fusion import TransformerFuserCustom


class PoseEstimator(nn.Module):
    """Pose estimator that uses custom transformer fusion between RGB and point features."""

    def __init__(self, num_objects=13, rgb_dim=32, point_dim=32, embed_dim=128, use_transformer=True):
        super().__init__()
        self.rgb_enc = RGBEncoder(out_dim=rgb_dim)
        self.point_enc = PointNetEncoder(out_dim=point_dim)
        self.use_transformer = use_transformer
        if use_transformer:
            self.fuser = TransformerFuserCustom(rgb_dim, point_dim, embed_dim=embed_dim, num_heads=2, num_layers=2)
            fused_dim = embed_dim
        else:
            fused_dim = rgb_dim + point_dim
            self.fuse_mlp = nn.Sequential(nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, fused_dim), nn.ReLU())

        self.pose_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 7)
        )
        self.conf_head = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rgb, points):
        # rgb: [B, 3, H, W], points: [B, N, 3]
        rgb_feat = self.rgb_enc(rgb)
        point_feat = self.point_enc(points)
        if self.use_transformer:
            fused = self.fuser(rgb_feat, point_feat)
        else:
            fused = torch.cat([rgb_feat, point_feat], dim=1)
            fused = self.fuse_mlp(fused)
        pose = self.pose_head(fused)
        translation = pose[:, :3]
        quat = pose[:, 3:]
        quat = F.normalize(quat, p=2, dim=1)
        pose = torch.cat([translation, quat], dim=1)
        conf = self.conf_head(fused)
        return pose, conf
