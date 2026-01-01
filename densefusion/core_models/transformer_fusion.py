import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """A small, from-scratch Multi-Head Self-Attention implementation.

    This implementation avoids `nn.Transformer` to make the attention
    computation explicit for reviewers.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, T, E]
        B, T, E = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3E]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, num_heads, T, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B, H, T, head_dim]
        context = context.transpose(1, 2).reshape(B, T, E)
        out = self.out_proj(context)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerFuserCustom(nn.Module):
    """Fuse two modality vectors (RGB + points) via a small transformer encoder.

    Input: rgb_features [B, C], point_features [B, C]
    Output: fused [B, embed_dim]
    """

    def __init__(self, rgb_dim, point_dim, embed_dim=128, num_heads=2, num_layers=2, dropout=0.0):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)
        self.point_proj = nn.Linear(point_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 3, embed_dim))
        self.layers = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_feat, point_feat):
        B = rgb_feat.shape[0]
        rgb_emb = self.rgb_proj(rgb_feat).unsqueeze(1)
        point_emb = self.point_proj(point_feat).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, rgb_emb, point_emb], dim=1)  # [B, 3, E]
        seq = seq + self.pos_embed
        for layer in self.layers:
            seq = layer(seq)
        seq = self.norm(seq)
        return seq[:, 0]  # return CLS token as fused vector
