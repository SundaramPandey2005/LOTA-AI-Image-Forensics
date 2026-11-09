import numpy as np
import pytest
import torch
from PIL import Image

from src.forensic.bitplanes import compose_low_bit_planes, extract_bit_planes
from src.forensic.lowbit import ensure_min_size, low_bit_noise_image
from src.forensic.mgps import gradient_scores, sample_candidate_patches, select_patches


def random_image(h=96, w=80, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestBitPlanes:
    def test_eq1_exact_reconstruction(self):
        image = random_image()
        planes = extract_bit_planes(image)
        rebuilt = sum(planes[..., k].astype(np.int64) << k for k in range(8))
        assert np.array_equal(rebuilt, image)

    def test_eq2_composition_matches_bit_mask(self):
        image = random_image()
        assert np.array_equal(compose_low_bit_planes(image, [0, 1, 2]), (image & 7).astype(np.float32))


class TestLowBitNoise:
    def test_scaling_matches_official_formula(self):
        image = random_image()
        noise = low_bit_noise_image(image, "scaling")
        assert noise.dtype == np.uint8 and noise.shape == image.shape
        assert np.array_equal(noise, (image & 0x07) * (255 // 7))
        assert noise.max() <= 252

    def test_thresholding_eq4(self):
        image = np.array([[[8, 9, 16], [7, 0, 255]]], dtype=np.uint8)
        noise = low_bit_noise_image(image, "thresholding")
        assert noise.tolist() == [[[0, 255, 0], [255, 0, 255]]]

    def test_rejects_float_images(self):
        with pytest.raises(TypeError):
            low_bit_noise_image(random_image().astype(np.float32))

    def test_resizing_rgb_first_changes_the_low_bits(self):
        """The reason bit-planes must be extracted at native resolution."""
        image = random_image(128, 128)
        native = low_bit_noise_image(image)
        resized_first = low_bit_noise_image(np.asarray(Image.fromarray(image).resize((64, 64), Image.BILINEAR)))
        downsampled_noise = native[::2, ::2]
        assert (resized_first != downsampled_noise).mean() > 0.5

    def test_small_images_are_upsampled_only_when_needed(self):
        assert ensure_min_size(random_image(40, 50), 32).shape == (40, 50, 3)
        assert ensure_min_size(random_image(20, 50), 32, 256).shape == (256, 256, 3)


class TestMGPS:
    def test_gradient_score_equals_eq5_convolutions(self):
        """The difference form equals the L1 norms of the four kernel responses."""
        patches = np.random.default_rng(1).integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        x = torch.from_numpy(patches).permute(0, 3, 1, 2).reshape(-1, 1, 32, 32).double()
        kernels = (
            torch.tensor([[-1.0, 1.0]]),                 # g_x
            torch.tensor([[-1.0], [1.0]]),               # g_y
            torch.tensor([[-1.0, 0.0], [0.0, 1.0]]),     # g_xy
            torch.tensor([[0.0, -1.0], [1.0, 0.0]]),     # g_yx
        )
        reference = sum(
            torch.nn.functional.conv2d(x, k.double()[None, None]).abs().reshape(5, -1).sum(dim=1) for k in kernels
        )
        assert np.array_equal(gradient_scores(patches), reference.numpy().astype(np.int64))

    def test_flat_patch_scores_zero(self):
        assert gradient_scores(np.full((1, 32, 32, 3), 77, dtype=np.uint8))[0] == 0

    def test_candidates_are_crops_of_the_noise_image(self):
        noise = low_bit_noise_image(random_image(100, 120))
        candidates = sample_candidate_patches(noise, 32, 64, np.random.default_rng(0))
        assert candidates.shape == (64, 32, 32, 3)

    def test_topk_is_sorted_and_contains_the_original_lota_patch(self):
        noise = low_bit_noise_image(random_image(200, 200, seed=3))
        top1, score1 = select_patches(noise, 1, rng=np.random.default_rng(7))
        top8, score8 = select_patches(noise, 8, rng=np.random.default_rng(7))
        assert np.all(np.diff(score8) <= 0)
        assert np.array_equal(top8[0], top1[0]) and score8[0] == score1[0]
        all_scores = gradient_scores(sample_candidate_patches(noise, 32, 64, np.random.default_rng(7)))
        assert score1[0] == all_scores.max()

    def test_min_selection_is_the_reverse_extreme(self):
        noise = low_bit_noise_image(random_image(200, 200, seed=4))
        _, low = select_patches(noise, 1, selection="min", rng=np.random.default_rng(2))
        _, high = select_patches(noise, 1, selection="max", rng=np.random.default_rng(2))
        assert low[0] <= high[0]

    def test_max_gradient_prefers_the_noisy_region(self):
        noise = np.zeros((128, 128, 3), dtype=np.uint8)
        noise[64:, 64:] = np.random.default_rng(0).integers(0, 8, (64, 64, 3), dtype=np.uint8) * 36
        patch, _ = select_patches(noise, 1, rng=np.random.default_rng(0))
        assert (patch[0] > 0).mean() > 0.5
