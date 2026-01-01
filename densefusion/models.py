import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config


class RGBFeatureExtractor(nn.Module):
    def __init__(self, d_rgb=32):
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
        self.final_conv = nn.Conv2d(512, d_rgb, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.psp(x)
        x = self.final_conv(x)
        return x


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, d_geo=32):
        super(PointNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, d_geo)
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


class TransformerFuser(nn.Module):
    def __init__(self, rgb_feature_dim, geo_feature_dim, embed_dim=config.TRANSFORMER_DIM, num_heads=config.TRANSFORMER_HEADS, num_layers=config.TRANSFORMER_LAYERS, dropout=config.TRANSFORMER_DROPOUT):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_feature_dim, embed_dim)
        self.point_proj = nn.Linear(geo_feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_features, point_features):
        batch_size = rgb_features.shape[0]
        rgb_embed = self.rgb_proj(rgb_features).unsqueeze(1)
        point_embed = self.point_proj(point_features).unsqueeze(1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        token_sequence = torch.cat((cls_tokens, rgb_embed, point_embed), dim=1)
        token_sequence = token_sequence + self.pos_embedding
        fused_sequence = self.transformer(token_sequence)
        fused_sequence = self.norm(fused_sequence)
        cls_output = fused_sequence[:, 0]
        return cls_output


class DenseFusionNetwork(nn.Module):
    def __init__(self, num_objects=13, use_transformer=config.USE_TRANSFORMER_FUSION, d_rgb=32, d_geo=32):
        super(DenseFusionNetwork, self).__init__()
        self.use_transformer = use_transformer
        self.rgb_extractor = RGBFeatureExtractor(d_rgb)
        self.point_extractor = PointNetFeatureExtractor(d_geo)
        if self.use_transformer:
            self.transformer_fuser = TransformerFuser(rgb_feature_dim=d_rgb, geo_feature_dim=d_geo)
            self.pose_fc1 = nn.Linear(config.TRANSFORMER_DIM, 512)
            self.pose_fc2 = nn.Linear(512, 256)
            self.pose_fc3 = nn.Linear(256, 7)
            self.conf_fc1 = nn.Linear(config.TRANSFORMER_DIM, 256)
            self.conf_fc2 = nn.Linear(256, 64)
            self.conf_fc3 = nn.Linear(64, 1)
        else:
            self.fusion_head = nn.Sequential(nn.Linear(d_rgb + d_geo, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU())
            self.pose_fc1 = nn.Linear(1024, 512)
            self.pose_fc2 = nn.Linear(512, 256)
            self.pose_fc3 = nn.Linear(256, 7)
            self.conf_fc1 = nn.Linear(1024, 256)
            self.conf_fc2 = nn.Linear(256, 64)
            self.conf_fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb, points):
        batch_size = rgb.size(0)
        rgb_features = self.rgb_extractor(rgb).view(batch_size, -1)
        point_features = self.point_extractor(points)
        if self.use_transformer:
            x = self.transformer_fuser(rgb_features, point_features)
        else:
            combined_features = torch.cat([rgb_features, point_features], dim=1)
            x = self.fusion_head(combined_features)
        pose_x = F.relu(self.pose_fc1(x))
        pose_x = self.dropout(pose_x)
        pose_x = F.relu(self.pose_fc2(pose_x))
        pose_x = self.dropout(pose_x)
        pose = self.pose_fc3(pose_x)
        translation = pose[:, :3]
        quaternion = pose[:, 3:]
        quaternion = F.normalize(quaternion, p=2, dim=1)
        pose = torch.cat([translation, quaternion], dim=1)
        conf_x = F.relu(self.conf_fc1(x))
        conf_x = self.dropout(conf_x)
        conf_x = F.relu(self.conf_fc2(conf_x))
        confidence = self.conf_fc3(conf_x)
        return pose, confidence
