from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


def get_directional_gradient_kernels(device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct the 4 directional 2D gradient convolution kernels defined in Eq. (5):
        d1 (horizontal): [-1, 1]          (1x2)
        d2 (vertical):   [-1; 1]          (2x1)
        d3 (45-degree):  [-1, 0; 0, 1]    (2x2)
        d4 (135-degree): [0, -1; 1, 0]    (2x2)
    """
    dev = device or torch.device("cpu")
    d1 = torch.tensor([[-1.0, 1.0]], dtype=dtype, device=dev).view(1, 1, 1, 2)
    d2 = torch.tensor([[-1.0], [1.0]], dtype=dtype, device=dev).view(1, 1, 2, 1)
    d3 = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=dtype, device=dev).view(1, 1, 2, 2)
    d4 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype, device=dev).view(1, 1, 2, 2)
    return d1, d2, d3, d4


def _get_gradient_kernels(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Padded to uniform (4, 1, 2, 2) kernels for fast batched 2D convolution.
    """
    kernels = torch.zeros((4, 1, 2, 2), dtype=dtype, device=device)
    kernels[0, 0, 0, 0] = -1.0
    kernels[0, 0, 0, 1] = 1.0
    kernels[1, 0, 0, 0] = -1.0
    kernels[1, 0, 1, 0] = 1.0
    kernels[2, 0, 0, 0] = -1.0
    kernels[2, 0, 1, 1] = 1.0
    kernels[3, 0, 0, 1] = -1.0
    kernels[3, 0, 1, 0] = 1.0
    return kernels


def compute_patch_divergence_scores(
    patches: torch.Tensor
) -> torch.Tensor:
    """
    Compute divergence score g_p for a collection of patches according to Eq. (5):
        g_p = ||z_p * g_x||_1 + ||z_p * g_y||_1 + ||z_p * g_xy||_1 + ||z_p * g_yx||_1

    Args:
        patches (torch.Tensor): Tensor of shape (Num_Patches, C, P_H, P_W) or (B, Num_Patches, C, P_H, P_W).

    Returns:
        torch.Tensor: Scalar score g_p per patch of shape (Num_Patches,) or (B, Num_Patches).
    """
    orig_ndim = patches.ndim
    if orig_ndim == 4: # (N, C, H, W)
        N, C, H, W = patches.shape
        x = patches.view(N * C, 1, H, W)
    elif orig_ndim == 5: # (B, N, C, H, W)
        B, N, C, H, W = patches.shape
        x = patches.view(B * N * C, 1, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {patches.shape}")

    kernels = _get_gradient_kernels(device=patches.device, dtype=patches.dtype) # (4, 1, 2, 2)
    grad_maps = F.conv2d(x, kernels, padding=0)
    l1_norms = torch.abs(grad_maps).sum(dim=(-3, -2, -1))

    if orig_ndim == 4:
        l1_norms = l1_norms.view(N, C).sum(dim=-1) # (N,)
    else:
        l1_norms = l1_norms.view(B, N, C).sum(dim=-1) # (B, N)

    return l1_norms


def maximum_gradient_patch_selection(
    noise_image: torch.Tensor,
    patch_size: int = 32,
    strategy: str = "max_gradient"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Maximum Gradient Patch Selection (MGPS) according to Section 3.2:
    Divides the 256x256 noise image into non-overlapping patches and selects
    the optimal patch according to divergence score g_p.
    """
    is_batched = (noise_image.ndim == 4)
    if not is_batched:
        noise_image = noise_image.unsqueeze(0) # (1, C, H, W)

    B, C, H, W = noise_image.shape
    num_h = H // patch_size
    num_w = W // patch_size
    num_patches = num_h * num_w

    patches = noise_image.view(B, C, num_h, patch_size, num_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, num_patches, C, patch_size, patch_size)

    if strategy == "max_gradient":
        scores = compute_patch_divergence_scores(patches) # (B, num_patches)
        best_indices = torch.argmax(scores, dim=-1) # (B,)
    elif strategy == "min_gradient":
        scores = compute_patch_divergence_scores(patches) # (B, num_patches)
        best_indices = torch.argmin(scores, dim=-1) # (B,)
    elif strategy == "random":
        best_indices = torch.randint(0, num_patches, (B,), device=noise_image.device)
    elif strategy == "center":
        center_h = num_h // 2
        center_w = num_w // 2
        center_idx = center_h * num_w + center_w
        best_indices = torch.full((B,), center_idx, dtype=torch.long, device=noise_image.device)
    else:
        raise ValueError(f"Unknown patch selection strategy: {strategy}")

    gather_idx = best_indices.view(B, 1, 1, 1, 1).expand(B, 1, C, patch_size, patch_size)
    selected_patches = torch.gather(patches, dim=1, index=gather_idx).squeeze(1)

    if not is_batched:
        selected_patches = selected_patches.squeeze(0)
        best_indices = best_indices.squeeze(0)

    return selected_patches, best_indices
