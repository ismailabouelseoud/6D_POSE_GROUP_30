# FILE: models/densefusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBFeatureExtractor(nn.Module):
    """RGB feature extraction network - PSPNet inspired"""
    def __init__(self):
        super(RGBFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.psp = nn.AdaptiveAvgPool2d((1, 1))
        self.final_conv = nn.Conv2d(512, 32, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.psp(x)
        x = self.final_conv(x)
        return x

class PointNetFeatureExtractor(nn.Module):
    """PointNet-style feature extractor for point clouds"""
    def __init__(self):
        super(PointNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SimpleTransformerFusion(nn.Module):
    """Simple transformer for fusing RGB and point cloud features"""
    def __init__(self, feature_dim=64, num_heads=4, num_layers=2, ffn_dim=256, dropout=0.1):
        super(SimpleTransformerFusion, self).__init__()
        self.feature_dim = feature_dim
        self.pos_encoding = nn.Parameter(torch.randn(1, 5000, feature_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            activation="gelu", dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(feature_dim, 1024)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        pos_enc = self.pos_encoding[:, :num_points, :]
        x = x + pos_enc
        x = self.transformer(x)
        x = self.output_proj(x)
        x = self.dropout(x)
        return x

class DenseFusionNetwork(nn.Module):
    """Dense fusion network combining RGB and point cloud features"""
    def __init__(self, num_objects, use_transformer, config):
        super(DenseFusionNetwork, self).__init__()
        self.num_objects = num_objects
        self.use_transformer = use_transformer
        self.rgb_extractor = RGBFeatureExtractor()
        self.point_extractor = PointNetFeatureExtractor()

        if self.use_transformer:
            print("Using Transformer Fusion")
            self.transformer_fusion = SimpleTransformerFusion(
                feature_dim=config.TRANSFORMER_DIM, num_heads=config.TRANSFORMER_HEADS,
                num_layers=config.TRANSFORMER_LAYERS, ffn_dim=config.TRANSFORMER_FFN_DIM,
                dropout=config.TRANSFORMER_DROPOUT
            )
        else:
            print("Using MLP Fusion")
            self.fusion_conv1 = nn.Conv1d(64, 256, 1)
            self.fusion_conv2 = nn.Conv1d(256, 512, 1)
            self.fusion_conv3 = nn.Conv1d(512, 1024, 1)
            self.fusion_bn1 = nn.BatchNorm1d(256)
            self.fusion_bn2 = nn.BatchNorm1d(512)
            self.fusion_bn3 = nn.BatchNorm1d(1024)

        self.pose_fc1 = nn.Linear(1024, 512)
        self.pose_fc2 = nn.Linear(512, 256)
        self.pose_fc3 = nn.Linear(256, 7)  # [tx, ty, tz, qw, qx, qy, qz]
        self.conf_fc1 = nn.Linear(1024, 256)
        self.conf_fc2 = nn.Linear(256, 64)
        self.conf_fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, points):
        batch_size = rgb.size(0)
        num_points = points.size(1)

        rgb_features = self.rgb_extractor(rgb).view(batch_size, 32)
        point_features = self.point_extractor(points)
        
        combined_features = torch.cat([rgb_features, point_features], dim=1)
        
        if self.use_transformer:
            transformer_input = combined_features.unsqueeze(1).repeat(1, num_points, 1)
            fused_features = self.transformer_fusion(transformer_input)
            x = torch.max(fused_features, dim=1)[0]
        else:
            combined_features_expanded = combined_features.unsqueeze(2).repeat(1, 1, num_points)
            x = F.relu(self.fusion_bn1(self.fusion_conv1(combined_features_expanded)))
            x = F.relu(self.fusion_bn2(self.fusion_conv2(x)))
            x = F.relu(self.fusion_bn3(self.fusion_conv3(x)))
            x = torch.max(x, 2)[0]

        pose_x = F.relu(self.pose_fc1(x))
        pose_x = self.dropout(pose_x)
        pose_x = F.relu(self.pose_fc2(pose_x))
        pose = self.pose_fc3(pose_x)
        
        translation = pose[:, :3]
        quaternion = F.normalize(pose[:, 3:], p=2, dim=1)
        pose = torch.cat([translation, quaternion], dim=1)

        conf_x = F.relu(self.conf_fc1(x))
        conf_x = self.dropout(conf_x)
        conf_x = F.relu(self.conf_fc2(conf_x))
        confidence = self.conf_fc3(conf_x)

        return pose, confidence