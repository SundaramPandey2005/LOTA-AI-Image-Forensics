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
