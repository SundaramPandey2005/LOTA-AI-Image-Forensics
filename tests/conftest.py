import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def tiny_genimage(tmp_path_factory) -> str:
    """A synthetic dataset in the GenImage layout (JPEG reals, PNG fakes)."""
    root = tmp_path_factory.mktemp("GenImage")
    rng = np.random.default_rng(0)
    for directory in ("imagenet_ai_0424_sdv5", "imagenet_ai_0419_biggan"):
        for split, count in (("train", 12), ("val", 6)):
            for cls in ("nature", "ai"):
                out = root / directory / split / cls
                out.mkdir(parents=True)
                for i in range(count):
                    if cls == "nature":
                        size = (int(rng.integers(64, 120)), int(rng.integers(64, 120)))
                        pixels = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(out / f"real_{i:03d}.JPEG", quality=90)
                    else:
                        pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(out / f"fake_{i:03d}.png")
    return str(root)
