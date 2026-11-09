"""Noise-Based Classifier (NBC) of LOTA, Sec. 3.3 -- the reproduced baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import get_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchClassifier(nn.Module):
    """ImageNet-pretrained ResNet-50 that scores a single low-bit patch.

    The 32x32 patch is bilinearly upsampled to 256x256, normalised with the
    ImageNet statistics and mapped to one real/fake logit (logit > 0 = AI).
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, input_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.net, feat_dim = get_backbone(backbone, pretrained)
        self.net.fc = nn.Linear(feat_dim, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, patches: torch.Tensor) -> torch.Tensor:
        """(N, 3, P, P) values in [0, 255] -> normalised (N, 3, S, S) float tensor."""
        x = patches.float() / 255.0
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(N, 3, P, P) -> logits of shape (N,)."""
        return self.net(self.preprocess(patches)).squeeze(-1)
