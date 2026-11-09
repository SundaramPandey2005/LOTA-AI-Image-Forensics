"""GenImage loading with the LOTA low-bit patch pipeline.

Expected layout (the official GenImage layout, also used by Tiny-GenImage):

    <root>/<generator_dir>/train/ai/*.png        fake images   (label 1)
    <root>/<generator_dir>/train/nature/*.JPEG   real images   (label 0)
    <root>/<generator_dir>/val/ai/*.png
    <root>/<generator_dir>/val/nature/*.JPEG

``generator_dir`` may be an official name such as ``imagenet_ai_0424_sdv5`` or
a short name such as ``sdv5``; both resolve to the same canonical key.
"""
import io
import os
import random
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset

from src.forensic.lowbit import ensure_min_size, low_bit_noise_image, resize_rgb
from src.forensic.mgps import select_patches

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
REAL_DIRS = ("nature", "0_real", "real")
FAKE_DIRS = ("ai", "1_fake", "fake")

# canonical key -> substrings that identify the generator directory
GENERATOR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "biggan": ("biggan",),
    "midjourney": ("midjourney",),
    "wukong": ("wukong",),
    "sdv4": ("sdv4", "sd14", "sdv1.4", "stable_diffusion_v_1_4", "stable_diffusion_v1.4"),
    "sdv5": ("sdv5", "sd15", "sdv1.5", "stable_diffusion_v_1_5", "stable_diffusion_v1.5"),
    "adm": ("adm",),
    "glide": ("glide",),
    "vqdm": ("vqdm",),
}
# Order used in the paper's tables.
CANONICAL_ORDER = ["biggan", "midjourney", "wukong", "sdv4", "sdv5", "adm", "glide", "vqdm"]


def canonical_name(dirname: str) -> str:
    low = dirname.lower()
    for key, aliases in GENERATOR_ALIASES.items():
        if any(alias in low for alias in aliases):
            return key
    return low


def _has_class_dirs(split_dir: str) -> bool:
    if not os.path.isdir(split_dir):
        return False
    children = {d.lower() for d in os.listdir(split_dir)}
    return any(d in children for d in REAL_DIRS) and any(d in children for d in FAKE_DIRS)


def discover_generators(root: str, max_depth: int = 3) -> Dict[str, str]:
    """Find generator directories under ``root``.

    A generator directory is any directory containing ``train`` and/or ``val``
    with both a real and a fake class folder. Returns {canonical_key: path},
    ordered as in the paper's tables.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset root '{os.path.abspath(root)}' does not exist. "
            f"See README.md (Dataset Preparation)."
        )
    found: Dict[str, str] = {}

    def visit(path: str, depth: int) -> None:
        if _has_class_dirs(os.path.join(path, "train")) or _has_class_dirs(os.path.join(path, "val")):
            key = canonical_name(os.path.basename(os.path.normpath(path)))
            found.setdefault(key, path)
            return
        if depth >= max_depth:
            return
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                visit(entry.path, depth + 1)

    visit(root, 0)
    ordered = {k: found[k] for k in CANONICAL_ORDER if k in found}
    ordered.update({k: v for k, v in sorted(found.items()) if k not in ordered})
    return ordered


def _class_dir(split_dir: str, names: Sequence[str]) -> Optional[str]:
    if not os.path.isdir(split_dir):
        return None
    for child in os.listdir(split_dir):
        if child.lower() in names:
            return os.path.join(split_dir, child)
    return None


def _scan_images(directory: Optional[str]) -> List[str]:
    if directory is None:
        return []
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def list_images(
    generator_dir: str,
    split: str,
    max_per_class: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[str, int]]:
    """List (path, label) pairs of one generator split; label 1 = AI-generated.

    ``max_per_class`` takes a deterministic random subset of each class.
    """
    split_dir = os.path.join(generator_dir, split)
    samples: List[Tuple[str, int]] = []
    for names, label in ((REAL_DIRS, 0), (FAKE_DIRS, 1)):
        files = _scan_images(_class_dir(split_dir, names))
        if max_per_class is not None and len(files) > max_per_class:
            random.Random(seed + label).shuffle(files)
            files = sorted(files[:max_per_class])
        samples.extend((f, label) for f in files)
    if not samples:
        raise RuntimeError(f"No images found under '{split_dir}'.")
    return samples


def split_train_val(
    samples: List[Tuple[str, int]],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Stratified, deterministic hold-out split of the training samples.

    The held-out part is used only for checkpoint selection, so the official
    GenImage ``val`` folders stay untouched as the test set.
    """
    train, val = [], []
    for label in (0, 1):
        subset = [s for s in samples if s[1] == label]
        random.Random(seed + 17 * label).shuffle(subset)
        n_val = int(round(len(subset) * val_fraction))
        val.extend(subset[:n_val])
        train.extend(subset[n_val:])
    return sorted(train), sorted(val)


def apply_perturbation(image: Image.Image, perturbation: Optional[str]) -> Image.Image:
    """Test-time degradations of Fig. 6: ``jpeg<quality>`` or ``blur<sigma>``."""
    if not perturbation or perturbation == "clean":
        return image
    if perturbation.startswith("jpeg"):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(perturbation[4:]))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if perturbation.startswith("blur"):
        return image.filter(ImageFilter.GaussianBlur(radius=float(perturbation[4:])))
    raise ValueError(f"Unknown perturbation '{perturbation}'")


class GenImagePatchDataset(Dataset):
    """Returns the top-``num_patches`` low-bit patches of every image.

    Pipeline per image (native resolution throughout):
        decode -> low-bit noise image -> ``num_candidates`` random crops ->
        gradient score -> keep the ``num_patches`` best, sorted by score.

    The patches are returned as uint8 so that upsampling and normalisation run
    batched on the GPU inside the model. ``patches[0]`` is always the single
    patch the original LOTA would have selected from the same candidates.

    Args:
        deterministic: if True, the candidate crops of an image depend only on
            its file name, so every model is evaluated on identical patches.
            If False (training), new candidates are drawn at every access.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        num_patches: int = 8,
        patch_size: int = 32,
        num_candidates: int = 64,
        bit_mode: str = "scaling",
        num_bits: int = 3,
        selection: str = "max",
        image_size: int = 256,
        deterministic: bool = False,
        perturbation: Optional[str] = None,
        resize_before_bits: bool = False,
    ):
        self.samples = list(samples)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_candidates = num_candidates
        self.bit_mode = bit_mode
        self.num_bits = num_bits
        self.selection = selection
        self.image_size = image_size
        self.deterministic = deterministic
        self.perturbation = perturbation
        self.resize_before_bits = resize_before_bits

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, path: str) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(zlib.crc32(os.path.basename(path).encode("utf-8")))
        # torch's RNG is seeded per worker and per epoch by the DataLoader.
        return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    def _load(self, path: str) -> np.ndarray:
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            image = apply_perturbation(image, self.perturbation)
            return np.asarray(image, dtype=np.uint8)
        except Exception as error:  # same fallback as the official loader
            print(f"[warn] unreadable image '{path}' ({error}); using a black image", flush=True)
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def extract(self, image: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """RGB uint8 image -> (patches (K, P, P, 3) uint8, scores (K,))."""
        if self.resize_before_bits:
            image = resize_rgb(image, self.image_size)
        noise = low_bit_noise_image(image, self.bit_mode, self.num_bits)
        noise = ensure_min_size(noise, self.patch_size, self.image_size)
        return select_patches(
            noise,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            num_candidates=self.num_candidates,
            selection=self.selection,
            rng=rng,
        )

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        patches, scores = self.extract(self._load(path), self._rng(path))
        patches = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2)  # (K, 3, P, P)
        return {
            "patches": patches,
            "scores": torch.from_numpy(scores.astype(np.float32)),
            "label": torch.tensor(float(label)),
            "index": index,
        }
