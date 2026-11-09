from typing import Tuple

import torch.nn as nn
import torchvision.models as models

# IMAGENET1K_V1 are the weights used by the official LOTA code (resnet50-19c8e357.pth).
_BACKBONES = {
    "resnet50": (models.resnet50, "IMAGENET1K_V1", 2048),
    "resnet18": (models.resnet18, "IMAGENET1K_V1", 512),
}


def get_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Return (torchvision backbone, feature dimension)."""
    name = name.lower()
    if name not in _BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Choose from {sorted(_BACKBONES)}.")
    builder, weights, feat_dim = _BACKBONES[name]
    return builder(weights=weights if pretrained else None), feat_dim
