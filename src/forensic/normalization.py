from typing import Union
import numpy as np
import torch


def normalize_noise_thresholding(
    noise_image: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Thresholding normalization according to Eq. (4):
        z_tilde_{i,j}^c = 0   if z_{i,j}^c == 0
        z_tilde_{i,j}^c = 255 if z_{i,j}^c > 0

    Binarizes and amplifies the sparse low-bit noise patterns to full [0, 255] range.

    Args:
        noise_image (np.ndarray or torch.Tensor): Composed low-bit noise map z^c.

    Returns:
        np.ndarray or torch.Tensor: Normalized noise map z_tilde with values in {0.0, 255.0}.
    """
    if isinstance(noise_image, torch.Tensor):
        return torch.where(noise_image > 0, torch.tensor(255.0, dtype=noise_image.dtype, device=noise_image.device), torch.tensor(0.0, dtype=noise_image.dtype, device=noise_image.device))
    elif isinstance(noise_image, np.ndarray):
        return np.where(noise_image > 0, 255.0, 0.0).astype(noise_image.dtype)
    else:
        raise TypeError(f"Unsupported noise image type: {type(noise_image)}")


def normalize_noise_scaling(
    noise_image: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Min-max scaling normalization according to Eq. (3):
        z_tilde^c = 255 * (z^c - z_min^c) / (z_max^c - z_min^c)

    Args:
        noise_image (np.ndarray or torch.Tensor): Composed low-bit noise map z^c.
        eps (float): Epsilon to prevent zero division.

    Returns:
        np.ndarray or torch.Tensor: Normalized noise map z_tilde scaled to [0.0, 255.0].
    """
    if isinstance(noise_image, torch.Tensor):
        # Per-channel / per-sample min-max normalization
        if noise_image.ndim == 4: # (B, C, H, W)
            z_min = noise_image.amin(dim=(-2, -1), keepdim=True)
            z_max = noise_image.amax(dim=(-2, -1), keepdim=True)
        elif noise_image.ndim == 3: # (C, H, W)
            z_min = noise_image.amin(dim=(-2, -1), keepdim=True)
            z_max = noise_image.amax(dim=(-2, -1), keepdim=True)
        else:
            z_min = noise_image.min()
            z_max = noise_image.max()
        denom = torch.clamp(z_max - z_min, min=eps)
        return 255.0 * (noise_image - z_min) / denom
    elif isinstance(noise_image, np.ndarray):
        if noise_image.ndim == 4: # (B, C, H, W) or (B, H, W, C)
            z_min = noise_image.min(axis=(-2, -1), keepdims=True)
            z_max = noise_image.max(axis=(-2, -1), keepdims=True)
        elif noise_image.ndim == 3:
            z_min = noise_image.min(axis=(0, 1), keepdims=True) if noise_image.shape[-1] == 3 else noise_image.min(axis=(-2, -1), keepdims=True)
            z_max = noise_image.max(axis=(0, 1), keepdims=True) if noise_image.shape[-1] == 3 else noise_image.max(axis=(-2, -1), keepdims=True)
        else:
            z_min = noise_image.min()
            z_max = noise_image.max()
        denom = np.maximum(z_max - z_min, eps)
        return (255.0 * (noise_image - z_min) / denom).astype(noise_image.dtype)
    else:
        raise TypeError(f"Unsupported noise image type: {type(noise_image)}")
