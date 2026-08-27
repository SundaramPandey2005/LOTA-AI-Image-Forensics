import os
import glob
from typing import List, Dict, Tuple, Optional, Any
import random
import numpy as np

# Standard 8 GenImage Generator IDs
ALL_GENIMAGE_GENERATORS = [
    "biggan",
    "sd14",
    "sd15",
    "midjourney",
    "adm",
    "glide",
    "wukong",
    "vqdm"
]

# 4 Key Representative Generators used for LOGO Benchmark
LOGO_REPRESENTATIVE_GENERATORS = [
    "biggan",
    "sd14",
    "midjourney",
    "adm"
]


class GenImageSplits:
    """Manager for generator dataset partitions."""
    def __init__(self, root_dir: str = "./data/GenImage", representative_only: bool = False):
        self.root_dir = root_dir
        self.generators = LOGO_REPRESENTATIVE_GENERATORS if representative_only else ALL_GENIMAGE_GENERATORS

    def get_logo_split(self, excluded_generator: str) -> Tuple[List[str], str]:
        train_gens = [g for g in self.generators if g != excluded_generator]
        return train_gens, excluded_generator


class LOGOPartitionManager:
    """Leave-One-Generator-Out partition manager."""
    def __init__(self, root_dir: str = "./data/GenImage", generators: Optional[List[str]] = None):
        self.root_dir = root_dir
        self.generators = generators or ALL_GENIMAGE_GENERATORS

    def get_partition(self, excluded_generator: str) -> Dict[str, List[str]]:
        seen_train = [g for g in self.generators if g != excluded_generator]
        return {
            "train": seen_train,
            "val": seen_train,
            "unseen_test": [excluded_generator],
            "cross_test": self.generators
        }


def scan_generator_directory(
    generator_dir: str,
    generator_name: str,
    max_samples_per_class: Optional[int] = None
) -> List[Dict[str, Any]]:
    samples = []
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPEG", "*.JPG", "*.PNG", "*.WEBP")

    real_paths = []
    for pattern in ["*real*/**/*", "*0_real*/**/*", "real/**/*", "*nature*/**/*", "nature/**/*", "*nature*/*", "nature/*"]:
        for ext in valid_exts:
            real_paths.extend(glob.glob(os.path.join(generator_dir, pattern, ext), recursive=True))
            real_paths.extend(glob.glob(os.path.join(generator_dir, pattern), recursive=True) if pattern.endswith(ext.replace("*", "")) else [])
    real_paths = sorted(list(set(real_paths)))

    fake_paths = []
    for pattern in ["*fake*/**/*", "*1_fake*/**/*", "*ai*/**/*", "fake/**/*", "ai/**/*", "*ai*/*", "ai/*"]:
        for ext in valid_exts:
            fake_paths.extend(glob.glob(os.path.join(generator_dir, pattern, ext), recursive=True))
    fake_paths = sorted(list(set(fake_paths)))

    if max_samples_per_class is not None:
        real_paths = real_paths[:max_samples_per_class]
        fake_paths = fake_paths[:max_samples_per_class]

    for p in real_paths:
        samples.append({"path": p, "label": 0, "generator": generator_name})

    for p in fake_paths:
        samples.append({"path": p, "label": 1, "generator": generator_name})

    return samples


def create_stratified_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    
    reals = [s for s in samples if s["label"] == 0]
    fakes = [s for s in samples if s["label"] == 1]

    rng.shuffle(reals)
    rng.shuffle(fakes)

    n_train_real = int(len(reals) * train_ratio)
    n_train_fake = int(len(fakes) * train_ratio)

    train_samples = reals[:n_train_real] + fakes[:n_train_fake]
    val_samples = reals[n_train_real:] + fakes[n_train_fake:]

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def create_logo_splits(
    all_generator_samples: Dict[str, List[Dict[str, Any]]],
    excluded_generator: str,
    train_val_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_candidates = []
    for gen_name, sample_list in all_generator_samples.items():
        if gen_name != excluded_generator:
            train_candidates.extend(sample_list)

    train_samples, val_samples = create_stratified_split(train_candidates, train_ratio=train_val_ratio, seed=seed)
    held_out_test_samples = all_generator_samples.get(excluded_generator, [])

    return train_samples, val_samples, held_out_test_samples
