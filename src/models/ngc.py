from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import get_backbone
from src.data.preprocessing import normalize_for_backbone


class NoiseGuidedAttention(nn.Module):
    """
    Spatial Cross-Attention module with Noise Guidance Bias E per Eq. (7):
        U = softmax( (Q * K^T) / sqrt(d_k) + E ) * V
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 8,
        patch_dim: int = 3 * 32 * 32, # 3072
        spatial_dim: int = 8 * 8       # 64
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.spatial_dim = spatial_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Guidance projection: maps flattened 32x32 noise patch (3072) to spatial bias E (num_heads x 64 x 64)
        self.guidance_proj = nn.Sequential(
            nn.Linear(patch_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads * spatial_dim * spatial_dim)
        )

    def forward(self, x_feat: torch.Tensor, noise_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_feat (torch.Tensor): Image spatial features of shape (B, N, C), where N=64, C=2048.
            noise_patch (torch.Tensor): Selected MGPS patch of shape (B, 3, 32, 32).
            
        Returns:
            torch.Tensor: Attended features of shape (B, N, C).
        """
        B, N, C = x_feat.shape

        # 1. Project Q, K, V
        q = self.q_proj(x_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D)
        k = self.k_proj(x_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D)
        v = self.v_proj(x_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D)

        # 2. Compute Guidance Bias E
        patch_flat = noise_patch.contiguous().view(B, -1) # (B, 3072)
        E = self.guidance_proj(patch_flat) # (B, H * N * N)
        E = E.view(B, self.num_heads, N, N)

        # 3. Scaled dot-product attention with forensic guidance bias (Eq. 7)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + E # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v) # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, C) # (B, N, C)
        out = self.out_proj(out)
        return out


class NoiseGuidedClassifier(nn.Module):
    """
    Noise-Guided Classifier (NGC) according to Section 3.4 of LOTA paper.
    
    Combines high-level semantic features from the raw image with micro-forensic
    spatial biases from the selected low-bit-plane noise patch via cross-attention.
    """
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        backbone: Optional[str] = None,
        pretrained: bool = True,
        num_classes: int = 1,
        num_heads: int = 8
    ):
        super().__init__()
        b_name = backbone or backbone_name or "resnet50"
        backbone_mod, feat_dim = get_backbone(b_name, pretrained=pretrained)
        
        self.conv1 = backbone_mod.conv1
        self.bn1 = backbone_mod.bn1
        self.relu = backbone_mod.relu
        self.maxpool = backbone_mod.maxpool
        
        self.layer1 = backbone_mod.layer1
        self.layer2 = backbone_mod.layer2
        self.layer3 = backbone_mod.layer3
        self.layer4 = backbone_mod.layer4
        
        # Spatial dimensions after layer4 for 256x256 input is 8x8 = 64
        self.attention = NoiseGuidedAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            patch_dim=3 * 32 * 32,
            spatial_dim=8 * 8
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def extract_image_features(self, raw_image: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map x_tilde from raw image."""
        x_norm = normalize_for_backbone(raw_image, input_is_255=True)
        
        out = self.conv1(x_norm)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # (B, 2048, 8, 8)
        
        B, C, H, W = out.shape
        # Flatten spatial dimensions to (B, H*W, C) -> (B, 64, 2048)
        feat = out.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return feat

    def forward(
        self,
        noise_patch: torch.Tensor,
        raw_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Dual-stream forward pass.
        
        Args:
            noise_patch (torch.Tensor): Selected MGPS patch (B, 3, 32, 32).
            raw_image (torch.Tensor): Input raw RGB image (B, 3, 256, 256).
            
        Returns:
            torch.Tensor: Binary logits of shape (B, 1) or (B,).
        """
        if raw_image is None:
            raise ValueError("NGC requires raw_image input in addition to noise_patch.")

        # 1. Extract semantic feature map from raw image
        feat = self.extract_image_features(raw_image) # (B, 64, 2048)

        # 2. Guided Cross-Attention with forensic bias E
        attn_out = self.attention(feat, noise_patch)
        x = self.norm1(feat + attn_out)
        
        # 3. FFN + Residual
        x = self.norm2(x + self.mlp(x)) # (B, 64, 2048)

        # 4. Global Average Pooling over 64 spatial tokens
        pooled = self.avgpool(x.transpose(1, 2)).squeeze(-1) # (B, 2048)
        logits = self.classifier(pooled) # (B, 1)
        
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


# Aliases
LOTAGuidedClassifier = NoiseGuidedClassifier
NGC = NoiseGuidedClassifier
