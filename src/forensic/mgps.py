"""Maximum Gradient Patch Selection (MGPS), LOTA Sec. 3.2, generalised to top-K.

Candidate patches are random crops of the native-resolution low-bit image, as
in the official LOTA loader. Each candidate gets the multi-directional gradient
score of Eq. (5); the original method keeps the single best patch (Eq. 6), the
proposed LOTA-MP keeps the K best.
"""
from typing import Optional, Tuple

import numpy as np


def sample_candidate_patches(
    noise: np.ndarray,
    patch_size: int = 32,
    num_candidates: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw ``num_candidates`` random ``patch_size`` crops from the low-bit image.

    The official code draws (256 // 32)^2 = 64 random crops (crops may overlap).

    Returns:
        uint8 array of shape (num_candidates, patch_size, patch_size, 3).
    """
    rng = rng or np.random.default_rng()
    h, w = noise.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Noise image {noise.shape} is smaller than the patch size {patch_size}")
    ys = rng.integers(0, h - patch_size + 1, size=num_candidates)
    xs = rng.integers(0, w - patch_size + 1, size=num_candidates)
    return np.stack([noise[y:y + patch_size, x:x + patch_size] for y, x in zip(ys, xs)])


def gradient_scores(patches: np.ndarray) -> np.ndarray:
    """Multi-directional gradient score g_p of Eq. (5) for a stack of patches.

    g_p = ||z * g_x||_1 + ||z * g_y||_1 + ||z * g_xy||_1 + ||z * g_yx||_1 with
    g_x = [-1 1], g_y = g_x^T, g_xy = [[-1 0], [0 1]], g_yx = [[0 -1], [1 0]],
    i.e. sums of absolute differences along the horizontal, vertical and both
    diagonal directions, accumulated over the three channels.

    Args:
        patches: array of shape (N, P, P, C).

    Returns:
        int64 array of shape (N,).
    """
    p = patches.astype(np.int32)
    g_x = np.abs(p[:, :, 1:] - p[:, :, :-1]).sum(axis=(1, 2, 3))
    g_y = np.abs(p[:, 1:, :] - p[:, :-1, :]).sum(axis=(1, 2, 3))
    g_xy = np.abs(p[:, 1:, 1:] - p[:, :-1, :-1]).sum(axis=(1, 2, 3))
    g_yx = np.abs(p[:, 1:, :-1] - p[:, :-1, 1:]).sum(axis=(1, 2, 3))
    return (g_x + g_y + g_xy + g_yx).astype(np.int64)


def select_patches(
    noise: np.ndarray,
    num_patches: int = 1,
    patch_size: int = 32,
    num_candidates: int = 64,
    selection: str = "max",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the ``num_patches`` best candidates of a low-bit image.

    ``num_patches=1`` with ``selection="max"`` is exactly the MGPS of the paper
    (Eq. 6). ``num_patches=K > 1`` returns the K highest-scoring candidates in
    descending score order, which is what the proposed LOTA-MP consumes. Because
    the output is sorted, the first patch is always the original LOTA patch.

    Args:
        selection: ``"max"`` (paper), ``"min"`` or ``"random"`` (ablations).

    Returns:
        (patches, scores): uint8 (K, P, P, 3) and int64 (K,).
    """
    if num_patches > num_candidates:
        raise ValueError("num_patches cannot exceed num_candidates")
    rng = rng or np.random.default_rng()
    candidates = sample_candidate_patches(noise, patch_size, num_candidates, rng)
    scores = gradient_scores(candidates)

    if selection == "max":
        order = np.argsort(-scores, kind="stable")
    elif selection == "min":
        order = np.argsort(scores, kind="stable")
    elif selection == "random":
        order = rng.permutation(num_candidates)
    else:
        raise ValueError(f"Unknown selection '{selection}' (use 'max', 'min' or 'random')")

    keep = order[:num_patches]
    return candidates[keep], scores[keep]
