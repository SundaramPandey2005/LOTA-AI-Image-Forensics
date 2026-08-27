from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(
    name: str = "resnet50",
    pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """
    Load a torchvision backbone network.
    
    Args:
        name (str): Backbone name ("resnet50", "resnet18", "resnet34", "resnet101").
        pretrained (bool): Whether to load standard ImageNet-1k pretrained weights.
        
    Returns:
        Tuple[nn.Module, int]: (backbone_module, feature_dim)
    """
    name = name.lower()
    weights = "DEFAULT" if pretrained else None

    if name == "resnet50":
        model = models.resnet50(weights=weights)
        feat_dim = 2048
    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        feat_dim = 512
    elif name == "resnet34":
        model = models.resnet34(weights=weights)
        feat_dim = 512
    elif name == "resnet101":
        model = models.resnet101(weights=weights)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model, feat_dim
