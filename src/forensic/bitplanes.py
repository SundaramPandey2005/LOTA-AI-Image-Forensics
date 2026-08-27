from typing import List, Union
import numpy as np
import torch


def extract_bit_planes(
    image: Union[np.ndarray, torch.Tensor],
    bits: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Decompose an 8-bit image channel into binary bit-planes according to Eq. (1):
        x^c = sum_{k=0}^7 2^k * x_k^c,  where x_k^c in {0, 1}

    Args:
        image (np.ndarray or torch.Tensor): Input uint8/float image with pixel values in [0, 255].
        bits (List[int]): List of bit indices to extract (0 = LSB, 7 = MSB).

    Returns:
        np.ndarray or torch.Tensor: Binary bit planes of shape (..., len(bits), H, W) or (H, W, C, len(bits)).
    """
    if isinstance(image, torch.Tensor):
        img_int = image.to(torch.int64)
        bit_planes = []
        for k in bits:
            bit_k = (img_int >> k) & 1
            bit_planes.append(bit_k)
        # Stack along a new dimension
        return torch.stack(bit_planes, dim=-1)
    elif isinstance(image, np.ndarray):
        img_int = image.astype(np.int64)
        bit_planes = []
        for k in bits:
            bit_k = (img_int >> k) & 1
            bit_planes.append(bit_k)
        return np.stack(bit_planes, axis=-1)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def compose_low_bit_planes(
    image: Union[np.ndarray, torch.Tensor],
    bit_indices: List[int] = [0, 1, 2]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Combine selected low-order bit-planes into a low-bit noise representation according to Eq. (2):
        z^c = sum_{i=0}^{K-1} 2^i * x_{k_i}^c

    For the default lowest 3 bit-planes (k=0, 1, 2):
        z^c = 4 * x_2^c + 2 * x_1^c + x_0^c  (values in 0..7)

    Args:
        image (np.ndarray or torch.Tensor): Input image with values in [0, 255].
            If torch.Tensor, shape is typically (B, C, H, W) or (C, H, W).
            If np.ndarray, shape is typically (H, W, C) or (C, H, W).
        bit_indices (List[int]): List of bit indices to combine. Default is [0, 1, 2].

    Returns:
        np.ndarray or torch.Tensor: Composed low-bit noise image with same spatial/channel layout.
    """
    if isinstance(image, torch.Tensor):
        img_int = image.to(torch.int64)
        z = torch.zeros_like(img_int, dtype=torch.float32)
        for i, k in enumerate(bit_indices):
            bit_k = (img_int >> k) & 1
            z += (2 ** i) * bit_k.to(torch.float32)
        return z
    elif isinstance(image, np.ndarray):
        img_int = image.astype(np.int64)
        z = np.zeros_like(img_int, dtype=np.float32)
        for i, k in enumerate(bit_indices):
            bit_k = (img_int >> k) & 1
            z += (2 ** i) * bit_k.astype(np.float32)
        return z
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
