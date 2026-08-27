import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GenImageDataset, GenImageForensicDataset, create_dataloader
from src.data.preprocessing import (
    load_and_preprocess_image,
    extract_lota_forensic_features,
    normalize_for_backbone,
    preprocess_raw_image,
    get_transforms,
    build_transforms,
)
from src.data.splits import (
    ALL_GENIMAGE_GENERATORS,
    LOGO_REPRESENTATIVE_GENERATORS,
    LOGOPartitionManager,
)


class TestDataModule:
    def test_mock_dataset_loading(self):
        """Verify mock dataset initialization and batch extraction."""
        train_ds = GenImageDataset(
            root_dir="./data/GenImage",
            generators=["sd15", "biggan"],
            split="train",
            use_mock_data=True,
            mock_num_samples=16,
        )
        assert len(train_ds) == 16
        sample = train_ds[0]
        assert "raw_image" in sample
        assert "noise_patch" in sample
        assert "label" in sample
        assert sample["noise_patch"].shape == (3, 32, 32)

    def test_dataloader_creation(self):
        """Verify create_dataloader utility with mock dataset."""
        train_ds = GenImageDataset(
            root_dir="./data/GenImage",
            generators=["sd15"],
            split="train",
            use_mock_data=True,
            mock_num_samples=8,
        )
        loader = create_dataloader(
            train_ds,
            batch_size=4,
            shuffle=True,
        )
        batch = next(iter(loader))
        assert batch["noise_patch"].shape == (4, 3, 32, 32)
        assert batch["label"].shape == (4,)

    def test_logo_partition_manager(self):
        """Verify LOGO partition manager generator exclusion and inclusion."""
        manager = LOGOPartitionManager(generators=LOGO_REPRESENTATIVE_GENERATORS)
        for holdout in LOGO_REPRESENTATIVE_GENERATORS:
            partition = manager.get_partition(holdout)
            assert partition["unseen_test"] == [holdout]
            assert holdout not in partition["train"]
            assert len(partition["train"]) == len(LOGO_REPRESENTATIVE_GENERATORS) - 1

    def test_forensic_feature_extraction_pipeline(self):
        """Verify extraction of bitplanes, normalization, and patch selection."""
        dummy_image = torch.randint(0, 256, (3, 256, 256), dtype=torch.float32)
        norm_noise, selected_patch, selected_idx = extract_lota_forensic_features(
            dummy_image,
            bit_indices=[0, 1, 2],
            normalization="thresholding",
            patch_size=32,
            patch_strategy="max_gradient",
        )
        assert norm_noise.shape == (3, 256, 256)
        assert selected_patch.shape == (3, 32, 32)
        assert 0 <= selected_idx < 64

    def test_no_silent_mock_fallback_on_missing_dir(self):
        """Verify that use_mock_data=False raises FileNotFoundError instead of silent fallback."""
        with pytest.raises(FileNotFoundError):
            GenImageDataset(
                root_dir="./data/NonExistentGenImageDir",
                generators=["biggan"],
                split="val",
                use_mock_data=False,
            )

    def test_pre_resize_preprocessing_disabled_by_default(self):
        """Verify that preprocessing remains identical when pre_resize_size is None or unset."""
        from PIL import Image
        import numpy as np

        np_arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_arr)

        out_default = preprocess_raw_image(pil_img, image_size=256)
        out_none = preprocess_raw_image(pil_img, image_size=256, pre_resize_size=None)

        assert out_default.shape == (3, 256, 256)
        assert out_none.shape == (3, 256, 256)
        assert torch.allclose(out_default, out_none)

    def test_pre_resize_preprocessing_enabled(self):
        """Verify that images are pre-resized to 128x128 and final tensor has shape (3, 256, 256)."""
        from PIL import Image
        import numpy as np

        np_arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_arr)

        out_pre_resized = preprocess_raw_image(pil_img, image_size=256, pre_resize_size=128)

        # Final tensor must still be (3, 256, 256)
        assert out_pre_resized.shape == (3, 256, 256)
        assert isinstance(out_pre_resized, torch.Tensor)
        assert out_pre_resized.dtype == torch.float32

        # Manually compute expected 512 -> 128 -> 256 transformation
        img_128 = pil_img.resize((128, 128), Image.BILINEAR)
        img_256 = img_128.resize((256, 256), Image.BILINEAR)
        expected_tensor = torch.from_numpy(np.array(img_256, dtype=np.float32)).permute(2, 0, 1)

        assert torch.allclose(out_pre_resized, expected_tensor, atol=1e-3)

    def test_dataset_pre_resize_parameter(self):
        """Verify GenImageDataset stores and defaults pre_resize_size appropriately."""
        ds_default = GenImageDataset(root_dir="./data/GenImage", use_mock_data=True)
        assert ds_default.pre_resize_size is None

        ds_128 = GenImageDataset(root_dir="./data/GenImage", pre_resize_size=128, use_mock_data=True)
        assert ds_128.pre_resize_size == 128

    def test_jpeg_reencode_preprocessing_disabled_by_default(self):
        """Verify that preprocessing is unchanged when jpeg_reencode_quality is None or unset."""
        from PIL import Image
        import numpy as np

        np_arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_arr)

        out_default = preprocess_raw_image(pil_img, image_size=256)
        out_none = preprocess_raw_image(pil_img, image_size=256, jpeg_reencode_quality=None)

        assert out_default.shape == (3, 256, 256)
        assert out_none.shape == (3, 256, 256)
        assert torch.allclose(out_default, out_none)

    def test_jpeg_reencode_preprocessing_enabled(self):
        """Verify that in-memory JPEG re-encoding applies DCT compression and outputs (3, 256, 256)."""
        import io
        from PIL import Image
        import numpy as np

        np_arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_arr)

        out_jpeg = preprocess_raw_image(pil_img, image_size=256, jpeg_reencode_quality=95)

        assert out_jpeg.shape == (3, 256, 256)
        assert isinstance(out_jpeg, torch.Tensor)
        assert out_jpeg.dtype == torch.float32

        # Manually compute in-memory JPEG compression
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        decoded = Image.open(buf).convert("RGB")
        expected_tensor = torch.from_numpy(np.array(decoded, dtype=np.float32)).permute(2, 0, 1)

        assert torch.allclose(out_jpeg, expected_tensor)

    def test_combined_resolution_and_encoding_matched_pipeline(self):
        """Verify the full 512x512 -> 128x128 -> JPEG Q95 -> 256x256 pipeline produces expected tensor."""
        import io
        from PIL import Image
        import numpy as np

        np_arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_arr)

        out_tensor = preprocess_raw_image(
            pil_img,
            image_size=256,
            pre_resize_size=128,
            jpeg_reencode_quality=95
        )

        assert out_tensor.shape == (3, 256, 256)

        # Expected sequential execution:
        # 1. Resize to 128x128
        img_128 = pil_img.resize((128, 128), Image.BILINEAR)
        # 2. In-memory JPEG encode & decode at Q95
        buf = io.BytesIO()
        img_128.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        img_encoded = Image.open(buf).convert("RGB")
        # 3. Resize to 256x256
        img_256 = img_encoded.resize((256, 256), Image.BILINEAR)
        expected = torch.from_numpy(np.array(img_256, dtype=np.float32)).permute(2, 0, 1)

        assert torch.allclose(out_tensor, expected, atol=1e-3)

    def test_dataset_jpeg_reencode_parameter(self):
        """Verify GenImageDataset stores and defaults jpeg_reencode_quality appropriately."""
        ds_default = GenImageDataset(root_dir="./data/GenImage", use_mock_data=True)
        assert ds_default.jpeg_reencode_quality is None

        ds_q95 = GenImageDataset(root_dir="./data/GenImage", jpeg_reencode_quality=95, use_mock_data=True)
        assert ds_q95.jpeg_reencode_quality == 95

