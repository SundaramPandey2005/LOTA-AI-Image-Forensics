"""Training objective for LOTA (K = 1) and LOTA-MP (K > 1)."""
import torch
import torch.nn.functional as F


def multi_patch_loss(
    image_logit: torch.Tensor,
    patch_logits: torch.Tensor,
    labels: torch.Tensor,
    instance_weight: float = 0.0,
) -> torch.Tensor:
    """L = BCE(s_bar, y) + lambda * (1 / K) * sum_k BCE(s_k, y).

    The first (bag) term trains exactly the decision rule used at inference.
    The optional second (instance) term supervises every patch individually;
    by Jensen's inequality it upper-bounds the bag term for ``logit_mean``.
    For K = 1 both terms coincide with the original LOTA BCE loss.
    """
    loss = F.binary_cross_entropy_with_logits(image_logit, labels)
    if instance_weight > 0:
        targets = labels.unsqueeze(1).expand_as(patch_logits)
        loss = loss + instance_weight * F.binary_cross_entropy_with_logits(patch_logits, targets)
    return loss
