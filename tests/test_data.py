import os

import torch

from src.data.genimage import (
    GenImagePatchDataset,
    apply_perturbation,
    canonical_name,
    discover_generators,
    list_images,
    split_train_val,
)


def test_canonical_names_cover_official_directories():
    assert canonical_name("imagenet_ai_0424_sdv5") == "sdv5"
    assert canonical_name("imagenet_ai_0419_sdv4") == "sdv4"
    assert canonical_name("imagenet_ai_0419_biggan") == "biggan"
    assert canonical_name("imagenet_midjourney") == "midjourney"
    assert canonical_name("imagenet_glide") == "glide"
    assert canonical_name("imagenet_ai_0508_adm") == "adm"


def test_discovery_and_labels(tiny_genimage):
    generators = discover_generators(tiny_genimage)
    assert list(generators) == ["biggan", "sdv5"]
    samples = list_images(generators["sdv5"], "train")
    assert len(samples) == 24
    assert sum(label for _, label in samples) == 12
    assert all((os.path.basename(os.path.dirname(path)) == "ai") == bool(label) for path, label in samples)


def test_subsampling_is_deterministic(tiny_genimage):
    directory = discover_generators(tiny_genimage)["sdv5"]
    assert list_images(directory, "train", 5) == list_images(directory, "train", 5)
    assert len(list_images(directory, "train", 5)) == 10


def test_train_val_split_is_disjoint_and_stratified(tiny_genimage):
    samples = list_images(discover_generators(tiny_genimage)["sdv5"], "train")
    train, val = split_train_val(samples, 0.25)
    assert not set(train) & set(val) and len(train) + len(val) == len(samples)
    assert sum(label for _, label in val) == 3 and len(val) == 6


def test_dataset_item_shapes(tiny_genimage):
    samples = list_images(discover_generators(tiny_genimage)["biggan"], "val")
    item = GenImagePatchDataset(samples, num_patches=4, deterministic=True)[0]
    assert item["patches"].shape == (4, 3, 32, 32) and item["patches"].dtype == torch.uint8
    assert item["scores"].shape == (4,) and torch.all(item["scores"][:-1] >= item["scores"][1:])


def test_evaluation_patches_are_reproducible_and_nested(tiny_genimage):
    """Every K sees the same candidates, so K=1 is a prefix of K=8."""
    samples = list_images(discover_generators(tiny_genimage)["biggan"], "val")
    k8 = GenImagePatchDataset(samples, num_patches=8, deterministic=True)
    k1 = GenImagePatchDataset(samples, num_patches=1, deterministic=True)
    for index in (0, len(samples) - 1):
        assert torch.equal(k8[index]["patches"], k8[index]["patches"])
        assert torch.equal(k8[index]["patches"][:1], k1[index]["patches"])


def test_training_patches_are_resampled(tiny_genimage):
    samples = list_images(discover_generators(tiny_genimage)["sdv5"], "train")
    dataset = GenImagePatchDataset(samples, num_patches=4, num_candidates=8, deterministic=False)
    torch.manual_seed(0)
    assert not torch.equal(dataset[0]["patches"], dataset[0]["patches"])


def test_perturbations_keep_size(tiny_genimage):
    from PIL import Image

    path, _ = list_images(discover_generators(tiny_genimage)["sdv5"], "val")[0]
    image = Image.open(path).convert("RGB")
    for name in ("jpeg90", "blur1", "clean", None):
        assert apply_perturbation(image, name).size == image.size
