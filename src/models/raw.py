from typing import Optional, Dict
import torch
import torch.nn as nn

from src.models.backbones import get_backbone
from src.data.preprocessing import normalize_for_backbone


class RawImageClassifier(nn.Module):
    """
    Standard Raw RGB Image Classifier (Sanity Check / Baseline).
    
    Uses the exact same ResNet-50 backbone architecture and classification head capacity
    as NBC and NGC, but operates purely on raw RGB images (without low-bit noise extraction,
    thresholding normalization, or MGPS patch selection).
    """
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        backbone: Optional[str] = None,
        pretrained: bool = True,
        num_classes: int = 1
    ):
        super().__init__()
        b_name = backbone or backbone_name or "resnet50"
        
        # Load backbone and extract feature dimensions
        backbone_mod, feat_dim = get_backbone(b_name, pretrained=pretrained)
        
        # Keep conv layers up to adaptive avgpool
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

    def extract_features(self, raw_image: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dim feature representation directly from raw RGB image.
        """
        # Apply standard ImageNet mean and std normalization
        x_norm = normalize_for_backbone(raw_image, input_is_255=True)
        
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
        raw_image: Optional[torch.Tensor] = None,
        noise_patch: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for Raw RGB baseline.
        
        Args:
            raw_image (torch.Tensor, optional): Input raw RGB image tensor (B, 3, 256, 256).
            noise_patch (torch.Tensor, optional): Ignored (for API uniformity).
            
        Returns:
            torch.Tensor: Binary logits of shape (B, 1) or (B,).
        """
        x = raw_image if raw_image is not None else kwargs.get("x", None)
        if x is None and noise_patch is not None and noise_patch.shape[-2:] == (256, 256):
            x = noise_patch
        elif x is None:
            raise ValueError("RawImageClassifier requires 'raw_image' input tensor.")

        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


# Aliases
RawOnlyClassifier = RawImageClassifier
RAW_ONLY = RawImageClassifier
