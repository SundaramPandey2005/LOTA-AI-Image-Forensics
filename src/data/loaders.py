"""DataLoader construction shared by training and evaluation."""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.genimage import GenImagePatchDataset


def build_dataset(
    samples: List[Tuple[str, int]],
    config: Dict[str, Any],
    num_patches: int,
    deterministic: bool,
    perturbation: Optional[str] = None,
) -> GenImagePatchDataset:
    forensic = config["forensic"]
    return GenImagePatchDataset(
        samples,
        num_patches=num_patches,
        patch_size=forensic["patch_size"],
        num_candidates=forensic["num_candidates"],
        bit_mode=forensic["bit_mode"],
        num_bits=forensic["num_bits"],
        selection=forensic["selection"],
        image_size=forensic["image_size"],
        deterministic=deterministic,
        perturbation=perturbation,
        resize_before_bits=forensic["resize_before_bits"],
    )


def build_loader(
    dataset: GenImagePatchDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=shuffle and num_workers > 0,  # training loader only
    )
