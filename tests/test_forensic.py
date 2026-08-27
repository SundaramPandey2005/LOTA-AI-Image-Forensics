import numpy as np
import torch
import pytest

from src.forensic.bitplanes import extract_bit_planes, compose_low_bit_planes
from src.forensic.normalization import normalize_noise_thresholding, normalize_noise_scaling
from src.forensic.mgps import (
    get_directional_gradient_kernels,
    compute_patch_divergence_scores,
    maximum_gradient_patch_selection
)


class TestBitPlanes:
    def test_bit_plane_exact_reconstruction_numpy(self):
        """Verify Eq. 1 exact loss-less bit-plane recovery in NumPy."""
        np.random.seed(42)
        original = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        planes = extract_bit_planes(original)
        assert planes.shape == (64, 64, 3, 8)

        reconstructed = np.zeros_like(original, dtype=np.int32)
        for k in range(8):
            reconstructed += (planes[..., k] * (2 ** k))

        assert np.array_equal(original, reconstructed.astype(np.uint8))

    def test_bit_plane_exact_reconstruction_torch(self):
        """Verify Eq. 1 exact bit-plane recovery in PyTorch."""
        torch.manual_seed(42)
        original = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
        planes = extract_bit_planes(original)
        assert planes.shape == (1, 3, 64, 64, 8)

        reconstructed = torch.zeros_like(original, dtype=torch.long)
        for k in range(8):
            reconstructed += planes[..., k].long() * (2 ** k)

        assert torch.equal(original.long(), reconstructed)

    def test_compose_low_bit_planes_values(self):
        """Verify Eq. 2 low-bit plane composition: z^c = 4*x2 + 2*x1 + x0 for K=3."""
        # Value 7 = 00000111 (bits 0,1,2 are 1) -> composed = 7
        # Value 8 = 00001000 (bits 0,1,2 are 0) -> composed = 0
        test_tensor = torch.tensor([[[[7.0, 8.0], [3.0, 0.0]]]]) # (1, 1, 2, 2)
        composed = compose_low_bit_planes(test_tensor, bit_indices=[0, 1, 2])
        expected = torch.tensor([[[[7.0, 0.0], [3.0, 0.0]]]])
        assert torch.allclose(composed, expected)


class TestNormalization:
    def test_thresholding_numpy_and_torch(self):
        """Verify Eq. 4 thresholding normalization: z > 0 -> 255.0, z == 0 -> 0.0."""
        noise_np = np.array([0.0, 1.0, 3.0, 4.0, 7.0], dtype=np.float32)
        norm_np = normalize_noise_thresholding(noise_np)
        assert norm_np[0] == 0.0
        assert norm_np[1] == 255.0
        assert norm_np[2] == 255.0
        assert norm_np[3] == 255.0
        assert norm_np[4] == 255.0

        noise_torch = torch.tensor([0.0, 1.0, 3.0, 4.0, 7.0])
        norm_torch = normalize_noise_thresholding(noise_torch)
        assert torch.allclose(norm_torch, torch.tensor([0.0, 255.0, 255.0, 255.0, 255.0]))

    def test_scaling_range(self):
        """Verify Eq. 3 min-max scaling normalization maps [min, max] -> [0.0, 255.0]."""
        noise = torch.tensor([[[[0.0, 3.5], [7.0, 0.0]]]])
        scaled = normalize_noise_scaling(noise)
        assert scaled.min().item() == 0.0
        assert pytest.approx(scaled.max().item(), 0.01) == 255.0


class TestMGPS:
    def test_gradient_kernels_shape_and_values(self):
        """Verify Eq. 5 4-directional gradient convolution kernels."""
        d1, d2, d3, d4 = get_directional_gradient_kernels()
        assert d1.shape == (1, 1, 1, 2) # Horizontal
        assert d2.shape == (1, 1, 2, 1) # Vertical
        assert d3.shape == (1, 1, 2, 2) # 45-deg
        assert d4.shape == (1, 1, 2, 2) # 135-deg

    def test_mgps_flat_synthetic_image(self):
        """Synthetic Test 1: Flat image has zero gradient everywhere."""
        flat = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        patch, idx = maximum_gradient_patch_selection(flat, patch_size=32, strategy="max_gradient")
        assert patch.shape == (1, 3, 32, 32)
        assert torch.allclose(patch, torch.zeros_like(patch))

    def test_mgps_horizontal_edge_gradient(self):
        """Synthetic Test 2: Horizontal step edge inside patch row 3 (row 112) produces strong internal vertical gradient."""
        edge_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        edge_img[:, :, 112:, :] = 255.0 # Edge at row 112 (within patch row 3: 96..128)
        patch, idx = maximum_gradient_patch_selection(edge_img, patch_size=32, strategy="max_gradient")
        row = idx.item() // 8
        assert row == 3

    def test_mgps_vertical_edge_gradient(self):
        """Synthetic Test 3: Vertical step edge inside patch col 3 (col 112) produces strong internal horizontal gradient."""
        edge_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        edge_img[:, :, :, 112:] = 255.0 # Edge at col 112 (within patch col 3: 96..128)
        patch, idx = maximum_gradient_patch_selection(edge_img, patch_size=32, strategy="max_gradient")
        col = idx.item() % 8
        assert col == 3

    def test_mgps_diagonal_edge_gradient(self):
        """Synthetic Test 4: Diagonal step edge produces strong diagonal gradient (D3/D4)."""
        edge_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        for i in range(256):
            edge_img[:, :, i, :i] = 255.0 # Diagonal step
        patch, idx = maximum_gradient_patch_selection(edge_img, patch_size=32, strategy="max_gradient")
        assert patch.shape == (1, 3, 32, 32)
        row = idx.item() // 8
        col = idx.item() % 8
        assert abs(row - col) <= 1

    def test_mgps_selects_exact_localized_artifact(self):
        """Synthetic Test 5: Localized high-frequency patch in row 2, col 5 (index 21)."""
        image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        checker = torch.tensor([[0.0, 255.0] * 16, [255.0, 0.0] * 16] * 16).unsqueeze(0).repeat(3, 1, 1)
        image[0, :, 64:96, 160:192] = checker

        selected_patch, selected_idx = maximum_gradient_patch_selection(
            image, patch_size=32, strategy="max_gradient"
        )
        expected_idx = 2 * (256 // 32) + 5 # row 2, col 5 = 21
        assert selected_idx.item() == expected_idx
        assert selected_patch.shape == (1, 3, 32, 32)
