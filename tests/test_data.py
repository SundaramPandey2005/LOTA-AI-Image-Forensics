import os
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

    def test_e4_config_validity_and_consistency(self):
        """Verify E4 multi-generator config exists and matches E1/E3 architecture and training parameters."""
        import os
        import yaml

        e4_path = "configs/multi_generator_biggan_vqdm_e4.yaml"
        e1_path = "configs/biggan_constrained_baseline_e1.yaml"
        e3_path = "configs/vqdm_e3_baseline.yaml"

        assert os.path.exists(e4_path), f"E4 config file not found at {e4_path}"
        assert os.path.exists(e1_path), f"E1 config file not found at {e1_path}"
        assert os.path.exists(e3_path), f"E3 config file not found at {e3_path}"

        with open(e4_path, "r") as f:
            cfg_e4 = yaml.safe_load(f)
        with open(e1_path, "r") as f:
            cfg_e1 = yaml.safe_load(f)
        with open(e3_path, "r") as f:
            cfg_e3 = yaml.safe_load(f)

        assert cfg_e4["experiment_name"] == "multi_generator_biggan_vqdm_e4"
        assert cfg_e4["data"]["generators"] == ["biggan", "vqdm"]
        assert cfg_e4["data"]["max_real_samples_per_generator"] == 250
        assert cfg_e4["data"]["max_fake_samples_per_generator"] == 250
        assert cfg_e4["data"]["max_real_samples"] == 500
        assert cfg_e4["data"]["max_fake_samples"] == 500

        # Verify model architecture parity with E1 and E3
        for k in ["architecture", "backbone", "pretrained", "num_classes"]:
            assert cfg_e4["model"][k] == cfg_e1["model"][k]
            assert cfg_e4["model"][k] == cfg_e3["model"][k]

        # Verify training hyperparameter parity with E1 and E3
        for k in ["batch_size", "epochs", "learning_rate", "weight_decay", "optimizer", "mixed_precision"]:
            assert cfg_e4["training"][k] == cfg_e1["training"][k]
            assert cfg_e4["training"][k] == cfg_e3["training"][k]

        # Verify reproducibility parity
        assert cfg_e4["reproducibility"]["seed"] == 42
        assert cfg_e4["reproducibility"]["deterministic"] is True

    def test_e4_mock_dataset_balanced_composition(self):
        """Verify E4 mock dataset produces exactly 1000 samples with 250 from each of the four categories."""
        from src.data.splits import get_multigen_category_counts

        ds = GenImageDataset(
            root_dir="./data/GenImage",
            generators=["biggan", "vqdm"],
            split="train",
            use_mock_data=True,
            mock_num_samples=1000,
        )

        assert len(ds) == 1000
        counts = get_multigen_category_counts(ds)

        assert counts["biggan"]["real"] == 250
        assert counts["biggan"]["fake"] == 250
        assert counts["vqdm"]["real"] == 250
        assert counts["vqdm"]["fake"] == 250

        # Verify total real and fake
        total_real = sum(1 for s in ds.samples if s[1] == 0.0)
        total_fake = sum(1 for s in ds.samples if s[1] == 1.0)
        assert total_real == 500
        assert total_fake == 500

    def test_e4_filesystem_dataset_composition_and_determinism(self):
        """Verify real-data discovery selects exactly 250 samples per category deterministically."""
        import tempfile
        from PIL import Image
        from src.data.splits import get_multigen_category_counts

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, "GenImage")
            # Populate mock directory structure with 500 real + 500 fake for BigGAN and VQDM (2000 files)
            for gen in ["biggan", "vqdm"]:
                for cat, label_dir in [("nature", "nature"), ("ai", "ai")]:
                    p = os.path.join(root, gen, "train", label_dir)
                    os.makedirs(p, exist_ok=True)
                    for i in range(500):
                        img = Image.new("RGB", (32, 32), color=(i % 256, (i * 3) % 256, (i * 7) % 256))
                        img.save(os.path.join(p, f"img_{i:04d}.png"))

            # Load with max_samples_per_class=250 for ["biggan", "vqdm"]
            ds1 = GenImageDataset(
                root_dir=root,
                generators=["biggan", "vqdm"],
                split="train",
                max_samples_per_class=250,
                use_mock_data=False,
                extract_forensics_on_the_fly=False
            )

            assert len(ds1) == 1000
            counts = get_multigen_category_counts(ds1)
            assert counts["biggan"]["real"] == 250
            assert counts["biggan"]["fake"] == 250
            assert counts["vqdm"]["real"] == 250
            assert counts["vqdm"]["fake"] == 250

            # Verify determinism: repeated discovery yields identical sample paths and order
            ds2 = GenImageDataset(
                root_dir=root,
                generators=["biggan", "vqdm"],
                split="train",
                max_samples_per_class=250,
                use_mock_data=False,
                extract_forensics_on_the_fly=False
            )
            assert ds1.samples == ds2.samples

    def test_e1_and_e3_dataset_behavior_unchanged(self):
        """Verify that single-generator baselines (E1 BigGAN, E3 VQDM) maintain exact 500/500 composition."""
        import tempfile
        from PIL import Image
        from src.data.splits import get_multigen_category_counts

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, "GenImage")
            for gen in ["biggan", "vqdm"]:
                for cat, label_dir in [("nature", "nature"), ("ai", "ai")]:
                    p = os.path.join(root, gen, "train", label_dir)
                    os.makedirs(p, exist_ok=True)
                    for i in range(500):
                        img = Image.new("RGB", (32, 32), color=(i % 256, 100, 100))
                        img.save(os.path.join(p, f"img_{i:04d}.png"))

            # E1 BigGAN-only
            ds_e1 = GenImageDataset(
                root_dir=root,
                generators=["biggan"],
                split="train",
                max_samples_per_class=500,
                use_mock_data=False,
                extract_forensics_on_the_fly=False
            )
            assert len(ds_e1) == 1000
            counts_e1 = get_multigen_category_counts(ds_e1)
            assert counts_e1["biggan"]["real"] == 500
            assert counts_e1["biggan"]["fake"] == 500
            assert "vqdm" not in counts_e1

            # E3 VQDM-only
            ds_e3 = GenImageDataset(
                root_dir=root,
                generators=["vqdm"],
                split="train",
                max_samples_per_class=500,
                use_mock_data=False,
                extract_forensics_on_the_fly=False
            )
            assert len(ds_e3) == 1000
            counts_e3 = get_multigen_category_counts(ds_e3)
            assert counts_e3["vqdm"]["real"] == 500
            assert counts_e3["vqdm"]["fake"] == 500
            assert "biggan" not in counts_e3

    def test_create_balanced_multigen_samples_helper(self):
        """Verify create_balanced_multigen_samples helper returns exactly balanced quotas."""
        from src.data.splits import create_balanced_multigen_samples, get_multigen_category_counts

        samples = create_balanced_multigen_samples(
            generators=["biggan", "vqdm"],
            samples_per_generator_class=250,
            use_mock_data=True
        )

        assert len(samples) == 1000
        counts = get_multigen_category_counts(samples)
        assert counts["biggan"]["real"] == 250
        assert counts["biggan"]["fake"] == 250
        assert counts["vqdm"]["real"] == 250
        assert counts["vqdm"]["fake"] == 250

    def test_e5_config_validity_and_consistency(self):
        """Verify E5 large-data config exists and maintains exact parity with E1 hyperparameters except sample limits."""
        import os
        import yaml

        e5_path = "configs/biggan_large_e5.yaml"
        e1_path = "configs/biggan_constrained_baseline_e1.yaml"

        assert os.path.exists(e5_path), f"E5 config file not found at {e5_path}"
        assert os.path.exists(e1_path), f"E1 config file not found at {e1_path}"

        with open(e5_path, "r") as f:
            cfg_e5 = yaml.safe_load(f)
        with open(e1_path, "r") as f:
            cfg_e1 = yaml.safe_load(f)

        assert cfg_e5["experiment_name"] == "biggan_large_e5"
        assert cfg_e5["generator"] == "biggan"
        assert cfg_e5["data"]["max_real_samples"] == 2000
        assert cfg_e5["data"]["max_fake_samples"] == 2000
        assert cfg_e5["data"]["train_val_ratio"] == 0.7
        assert cfg_e5["data"]["require_exact_sample_counts"] is True

        # Verify model architecture parity with E1
        for k in ["architecture", "backbone", "pretrained", "num_classes"]:
            assert cfg_e5["model"][k] == cfg_e1["model"][k]

        # Verify preprocessing parity with E1
        for k in ["image_size", "patch_size", "bit_planes", "normalization"]:
            assert cfg_e5["data"][k] == cfg_e1["data"][k]

        # Verify training hyperparameter parity with E1
        for k in ["batch_size", "epochs", "learning_rate", "weight_decay", "optimizer", "mixed_precision"]:
            assert cfg_e5["training"][k] == cfg_e1["training"][k]

        # Verify reproducibility parity
        assert cfg_e5["reproducibility"]["seed"] == 42
        assert cfg_e5["reproducibility"]["deterministic"] is True

    def test_e5_dataset_split_sample_counts(self):
        """Verify that 4000 raw samples (2000 real, 2000 fake) split into exactly 2800 train and 1200 val with 50/50 balance."""
        from src.data.splits import create_stratified_split

        raw_samples = []
        for i in range(2000):
            raw_samples.append({"path": f"/fake/path/real_{i}.png", "label": 0, "generator": "biggan"})
        for i in range(2000):
            raw_samples.append({"path": f"/fake/path/fake_{i}.png", "label": 1, "generator": "biggan"})

        assert len(raw_samples) == 4000

        train_samples, val_samples = create_stratified_split(raw_samples, train_ratio=0.7, seed=42)

        # Train split verification
        assert len(train_samples) == 2800
        train_reals = sum(1 for s in train_samples if s["label"] == 0)
        train_fakes = sum(1 for s in train_samples if s["label"] == 1)
        assert train_reals == 1400
        assert train_fakes == 1400

        # Val split verification
        assert len(val_samples) == 1200
        val_reals = sum(1 for s in val_samples if s["label"] == 0)
        val_fakes = sum(1 for s in val_samples if s["label"] == 1)
        assert val_reals == 600
        assert val_fakes == 600



