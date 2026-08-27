from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import get_backbone
from src.data.preprocessing import normalize_for_backbone


class NoiseBasedClassifier(nn.Module):
    """
    Noise-Based Classifier (NBC) according to Section 3.3 of the LOTA paper.
    
    1. Takes selected MGPS noise patch z_tilde_p* (shape B, 3, 32, 32 in [0, 255]).
    2. Upsamples patch bilinearly from 32x32 to 256x256.
    3. Applies ImageNet normalization.
    4. Extracts features using a pretrained ResNet-50 backbone.
    5. Outputs binary classification logit (0 = Real, 1 = AI-Generated).
    """
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        backbone: Optional[str] = None,
        pretrained: bool = True,
        num_classes: int = 1,
        target_size: int = 256
    ):
        super().__init__()
        self.target_size = target_size
        b_name = backbone or backbone_name or "resnet50"
        
        # Load backbone and replace its final fc layer with our linear head
        backbone_mod, feat_dim = get_backbone(b_name, pretrained=pretrained)
        
        # Keep conv layers up to avgpool
        self.conv1 = backbone_mod.conv1
        self.bn1 = backbone_mod.bn1
        self.relu = backbone_mod.relu
        self.maxpool = backbone_mod.maxpool
        
        self.layer1 = backbone_mod.layer1
        self.layer2 = backbone_mod.layer2
        self.layer3 = backbone_mod.layer3
        self.layer4 = backbone_mod.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_dim, num_classes)

    def extract_features(self, x_patch: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dim feature representation from noise patch.
        """
        # Upsample 32x32 patch to 256x256
        if x_patch.shape[-2:] != (self.target_size, self.target_size):
            x_up = F.interpolate(
                x_patch,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            )
        else:
            x_up = x_patch

        # Normalize with ImageNet stats
        x_norm = normalize_for_backbone(x_up, input_is_255=True)
        
        out = self.conv1(x_norm)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        pooled = self.avgpool(out)
        feat = torch.flatten(pooled, 1)
        return feat

    def forward(
        self,
        noise_patch: torch.Tensor,
        raw_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            noise_patch (torch.Tensor): Forensic patch tensor of shape (B, 3, 32, 32).
            raw_image (torch.Tensor, optional): Ignored by NBC (provided for uniform API).
            
        Returns:
            torch.Tensor: Binary logits of shape (B, 1) or (B,).
        """
        feat = self.extract_features(noise_patch)
        logits = self.fc(feat)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


# Aliases
LOTANoiseClassifier = NoiseBasedClassifier
NBC = NoiseBasedClassifier
