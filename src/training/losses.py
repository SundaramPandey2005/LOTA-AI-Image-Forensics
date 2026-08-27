import torch
import torch.nn as nn
from typing import Optional


def get_loss_function(
    loss_type: str = "bce_with_logits",
    pos_weight: Optional[float] = None
) -> nn.Module:
    """
    Build loss function for binary AI image classification.

    Args:
        loss_type (str): 'bce_with_logits' or 'bce'.
        pos_weight (float, optional): Weight for positive class.

    Returns:
        nn.Module: PyTorch loss module.
    """
    pos_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None

    if loss_type == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
    elif loss_type == "bce":
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Alias
build_loss_fn = get_loss_function
