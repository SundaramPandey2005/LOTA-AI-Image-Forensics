import os
import glob
from typing import List, Dict, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from src.data.preprocessing import get_transforms, preprocess_raw_image, extract_lota_forensic_features
from src.forensic.bitplanes import compose_low_bit_planes
from src.forensic.normalization import normalize_noise_thresholding, normalize_noise_scaling
from src.forensic.mgps import maximum_gradient_patch_selection


class GenImageDataset(Dataset):
    """
    PyTorch Dataset for GenImage with native forensic low-bit plane pipeline support.
    
    Supports:
    - Multi-generator training (e.g. for LOGO training sets).
    - Single-generator evaluation.
    - Deterministic subsampling (for pilot checks and compute-constrained testing).
    - Explicit mock dataset mode for unit testing and CI only (use_mock_data=True).
    
    STRICT RESEARCH INTEGRITY POLICY:
    - If use_mock_data=False (the default), this class will NEVER silently fall back to synthetic data.
    - Missing directories, missing files, or corrupt images raise explicit errors.
    """
    def __init__(
        self,
        root_dir: str = "./data/GenImage",
        generators: Union[str, List[str]] = "sd15",
        split: str = "train",
        image_size: int = 256,
        pre_resize_size: Optional[int] = None,
        jpeg_reencode_quality: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
        max_samples_per_class: Optional[Union[int, Dict[str, Any]]] = None,
        max_samples_per_generator_class: Optional[int] = None,
        extract_forensics_on_the_fly: bool = True,
        bit_planes: Optional[List[int]] = None,
        bit_indices: Optional[List[int]] = None,
        normalization_method: str = "thresholding",
        normalization: Optional[str] = None,
        patch_size: int = 32,
        patch_selection_strategy: str = "max_gradient",
        patch_strategy: Optional[str] = None,
        use_mock_data: bool = False,
        mock_num_samples: int = 64,
        samples: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.generators = [generators] if isinstance(generators, str) else list(generators)
        self.split = split
        self.image_size = image_size
        self.pre_resize_size = pre_resize_size
        self.jpeg_reencode_quality = jpeg_reencode_quality or jpeg_quality
        self.extract_forensics_on_the_fly = extract_forensics_on_the_fly
        self.bit_planes = bit_planes or bit_indices or [0, 1, 2]
        self.normalization_method = normalization or normalization_method or "thresholding"
        self.patch_size = patch_size
        self.patch_selection_strategy = patch_strategy or patch_selection_strategy or "max_gradient"
        self.use_mock_data = use_mock_data
        self.max_samples_per_class = max_samples_per_class
        self.max_samples_per_generator_class = max_samples_per_generator_class

        self.samples: List[Tuple[str, float, str]] = []

        if samples is not None:
            # Loaded from explicit sample dictionary list
            for s in samples:
                self.samples.append((s["path"], float(s["label"]), s.get("generator", "unknown")))
        elif self.use_mock_data:
            # Explicit mock mode for infrastructure/CI testing
            # When mock_num_samples is specified, evenly divide across generators
            # ensuring balanced real (0.0) and fake (1.0) samples per generator
            samples_per_gen = mock_num_samples // len(self.generators)
            for gen in self.generators:
                for i in range(samples_per_gen):
                    label = float(i % 2)
                    self.samples.append((f"mock_{gen}_{split}_{i}.png", label, gen))
            remainder = mock_num_samples % len(self.generators)
            for j in range(remainder):
                gen = self.generators[j]
                self.samples.append((f"mock_{gen}_{split}_{samples_per_gen + j}.png", float(j % 2), gen))
        else:
            # Real dataset mode: must exist and have samples
            if not os.path.exists(root_dir):
                raise FileNotFoundError(
                    f"GenImage dataset directory was not found at '{os.path.abspath(root_dir)}'.\n"
                    f"Real-data training cannot proceed without legitimate image files.\n"
                    f"If you intentionally wish to test software infrastructure with synthetic data, set use_mock_data=True."
                )
            quota = max_samples_per_generator_class if max_samples_per_generator_class is not None else max_samples_per_class
            self._discover_samples(quota)

    def _discover_samples(self, max_samples_per_class: Optional[Union[int, Dict[str, Any]]]):
        real_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")
        discovered_by_gen = {}
        seen_files = set()

        for gen in self.generators:
            discovered_by_gen[gen] = {"nature": 0, "ai": 0}
            for label_aliases, label_val in [
                (["nature", "0_real", "real"], 0.0),
                (["ai", "1_fake", "fake"], 1.0)
            ]:
                class_files = []
                for label_name in label_aliases:
                    dir_options = [
                        os.path.join(self.root_dir, gen, self.split, label_name),
                        os.path.join(self.root_dir, gen, label_name)
                    ]
                    for p_dir in dir_options:
                        if os.path.exists(p_dir):
                            for ext in real_exts:
                                for fp in glob.glob(os.path.join(p_dir, ext)):
                                    if fp not in seen_files:
                                        seen_files.add(fp)
                                        class_files.append(fp)
                                for fp in glob.glob(os.path.join(p_dir, "**", ext), recursive=True):
                                    if fp not in seen_files:
                                        seen_files.add(fp)
                                        class_files.append(fp)

                class_files = sorted(list(set(class_files)))

                # Determine limit for this generator and class
                limit = None
                if isinstance(max_samples_per_class, int):
                    limit = max_samples_per_class
                elif isinstance(max_samples_per_class, dict):
                    if gen in max_samples_per_class and isinstance(max_samples_per_class[gen], dict):
                        k = "nature" if label_val == 0.0 else "ai"
                        limit = max_samples_per_class[gen].get(k, max_samples_per_class[gen].get("real" if label_val == 0.0 else "fake"))
                    elif gen in max_samples_per_class:
                        limit = max_samples_per_class[gen]
                    elif "real" in max_samples_per_class or "fake" in max_samples_per_class:
                        k = "real" if label_val == 0.0 else "fake"
                        limit = max_samples_per_class.get(k)
                    elif "nature" in max_samples_per_class or "ai" in max_samples_per_class:
                        k = "nature" if label_val == 0.0 else "ai"
                        limit = max_samples_per_class.get(k)

                if limit is not None and limit > 0:
                    class_files = class_files[:limit]

                key = "nature" if label_val == 0.0 else "ai"
                discovered_by_gen[gen][key] = len(class_files)
                for fp in class_files:
                    self.samples.append((fp, label_val, gen))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image samples discovered under '{os.path.abspath(self.root_dir)}' "
                f"for generators {self.generators} and split '{self.split}'.\n"
                f"Breakdown: {discovered_by_gen}\n"
                f"Ensure that real nature images and fake AI images exist under the generator directories."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_synthetic_image(self, label: float) -> torch.Tensor:
        base = torch.randint(20, 235, (3, self.image_size, self.image_size), dtype=torch.float32)
        if label == 1.0:
            patch_noise = torch.randint(0, 8, (3, 64, 64), dtype=torch.float32)
            base[:, 32:96, 32:96] = (base[:, 32:96, 32:96].long() & ~7 | patch_noise.long()).float()
        return base

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str]]:
        file_path, label, generator = self.samples[idx]

        if self.use_mock_data:
            raw_tensor = self._generate_synthetic_image(label)
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file '{file_path}' does not exist.")
            try:
                pil_img = Image.open(file_path).convert("RGB")
                raw_tensor = preprocess_raw_image(
                    pil_img,
                    image_size=self.image_size,
                    pre_resize_size=self.pre_resize_size,
                    jpeg_reencode_quality=self.jpeg_reencode_quality
                )
            except Exception as e:
                raise RuntimeError(f"Corrupt or unreadable image file '{file_path}': {e}")

        sample = {
            "raw_image": raw_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": generator,
            "path": file_path
        }

        if self.extract_forensics_on_the_fly:
            norm_noise, selected_patch, patch_idx = extract_lota_forensic_features(
                raw_tensor,
                bit_indices=self.bit_planes,
                normalization=self.normalization_method,
                patch_size=self.patch_size,
                patch_strategy=self.patch_selection_strategy
            )
            sample["noise_patch"] = selected_patch
            sample["noise_image"] = norm_noise
            sample["patch_idx"] = patch_idx

        return sample


# Alias for compatibility
GenImageForensicDataset = GenImageDataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
