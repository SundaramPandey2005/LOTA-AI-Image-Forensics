"""Bit-Planes Guided Noisy Image Generation (BGNIG), LOTA Sec. 3.1.

The low-bit image is computed on the *decoded image at its native resolution*.
Any interpolation of the RGB image before this step re-computes the low-order
bits from weighted neighbour averages and destroys the signal LOTA relies on,
so resizing is never applied before bit extraction.
"""
from typing import Optional

import numpy as np
from PIL import Image


def low_bit_noise_image(
    image: np.ndarray,
    bit_mode: str = "scaling",
    num_bits: int = 3,
) -> np.ndarray:
    """Compose the ``num_bits`` lowest bit-planes (Eq. 2) and normalise them.

    Args:
        image: uint8 RGB array of shape (H, W, 3) at native resolution.
        bit_mode: ``"scaling"`` (Eq. 3, as in the official code) or
            ``"thresholding"`` (Eq. 4).
        num_bits: number of low-order bit-planes to keep (paper default: 3).

    Returns:
        uint8 noise image of shape (H, W, 3) with values in [0, 255].
    """
    if image.dtype != np.uint8:
        raise TypeError(f"Expected a uint8 image, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) image, got shape {image.shape}")
    if not 1 <= num_bits <= 8:
        raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")

    mask = (1 << num_bits) - 1
    z = image & np.uint8(mask)  # Eq. (2): z = sum_{k < num_bits} 2^k x_k

    if bit_mode == "scaling":
        # Official code: (img & 0x07) * (255 // 7) -> values in {0, 36, ..., 252}
        return (z * np.uint8(255 // mask)).astype(np.uint8)
    if bit_mode == "thresholding":
        return np.where(z > 0, np.uint8(255), np.uint8(0)).astype(np.uint8)
    raise ValueError(f"Unknown bit_mode '{bit_mode}' (use 'scaling' or 'thresholding')")


def ensure_min_size(noise: np.ndarray, patch_size: int, image_size: int = 256) -> np.ndarray:
    """Resize the noise image only if it is too small to crop a patch from.

    Mirrors the official loader, which resizes the low-bit image to
    ``image_size`` x ``image_size`` only when ``min(H, W) < patch_size``.
    """
    h, w = noise.shape[:2]
    if min(h, w) >= patch_size:
        return noise
    resized = Image.fromarray(noise).resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(resized)


def resize_rgb(image: np.ndarray, size: Optional[int]) -> np.ndarray:
    """Bilinear resize of the RGB image (only used by the ablation that
    reproduces the earlier, incorrect resize-before-bit-extraction order)."""
    if size is None:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))
