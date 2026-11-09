"""LOTA-MP: Multi-Patch Evidence Aggregation -- the PROPOSED MODIFICATION.

Original LOTA classifies an image from the single 32x32 low-bit patch with the
highest gradient score, i.e. from roughly 1.5% of a 256x256 image. LOTA-MP
keeps the K highest-scoring patches, scores each with the *same* shared
backbone and fuses the K log-odds into one image-level decision:

    s_bar(x) = (1 / K) * sum_k  f_theta(z_(k)),      p(fake | x) = sigmoid(s_bar)

Averaging log-odds is the product-of-experts / naive-Bayes fusion rule: if the
patches were conditionally independent given the class, the sum of per-patch
log-likelihood ratios is the Bayes-optimal statistic, and with balanced priors
its sign equals the sign of the mean. If each patch logit has class-conditional
mean +-mu, variance sigma^2 and pairwise correlation rho, then

    Var[s_bar] = sigma^2 * (rho + (1 - rho) / K)

so the decision margin mu / std grows by sqrt(K / (1 + (K - 1) * rho)).

With K = 1 this module is exactly the original LOTA noise-based classifier, so
baseline and proposed checkpoints share one format. No parameters are added.
"""
from typing import Tuple

import torch
import torch.nn as nn

from src.models.patch_classifier import PatchClassifier

AGGREGATIONS = ("logit_mean", "prob_mean", "vote")


def aggregate_logits(patch_logits: torch.Tensor, aggregation: str = "logit_mean") -> torch.Tensor:
    """Fuse per-patch logits (B, K) into one image-level logit (B,).

    ``logit_mean`` is the proposed rule. ``prob_mean`` (average probability)
    and ``vote`` (fraction of patches voting fake) are provided as ablations;
    both are mapped back to logit space so every rule shares the 0 threshold.
    """
    if patch_logits.dim() != 2:
        raise ValueError(f"Expected (B, K) logits, got {tuple(patch_logits.shape)}")
    if aggregation == "logit_mean":
        return patch_logits.mean(dim=1)
    if aggregation == "prob_mean":
        prob = torch.sigmoid(patch_logits).mean(dim=1)
    elif aggregation == "vote":
        prob = (patch_logits > 0).float().mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'. Choose from {AGGREGATIONS}.")
    prob = prob.clamp(1e-6, 1 - 1e-6)
    return torch.log(prob) - torch.log1p(-prob)


class MultiPatchLOTA(nn.Module):
    """Shared-backbone multi-patch detector (K = 1 reduces to original LOTA)."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_patches: int = 1,
        aggregation: str = "logit_mean",
        input_size: int = 256,
    ):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation '{aggregation}'. Choose from {AGGREGATIONS}.")
        self.num_patches = num_patches
        self.aggregation = aggregation
        self.classifier = PatchClassifier(backbone, pretrained, input_size)

    def patch_logits(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, K, 3, P, P) -> per-patch logits (B, K), one shared forward pass."""
        b, k = patches.shape[:2]
        return self.classifier(patches.flatten(0, 1)).view(b, k)

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uses the first ``num_patches`` (highest-scoring) patches.

        Returns:
            (image_logit (B,), patch_logits (B, K))
        """
        logits = self.patch_logits(patches[:, : self.num_patches])
        return aggregate_logits(logits, self.aggregation), logits
