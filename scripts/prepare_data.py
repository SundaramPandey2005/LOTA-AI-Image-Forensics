"""Locate / download GenImage and verify its layout.

    # verify an existing copy and print the per-generator inventory
    python scripts/prepare_data.py --data-root data/GenImage

    # download Tiny-GenImage (8.4 GB, Kaggle) and verify it
    python scripts/prepare_data.py --download --data-root data/GenImage

The script prints the resolved data root on its last line (``DATA_ROOT=...``)
so that notebooks can pick it up. No pre-processing is required: bit-planes are
extracted on the fly from the original files, which must not be resized or
re-encoded.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.genimage import discover_generators, list_images  # noqa: E402

KAGGLE_DATASET = "yangsangtai/tiny-genimage"
KAGGLE_INPUT_DIRS = ("/kaggle/input/tiny-genimage", "/kaggle/input/datasets/yangsangtai/tiny-genimage")


def download() -> str:
    for candidate in KAGGLE_INPUT_DIRS:  # already attached to a Kaggle notebook
        if os.path.isdir(candidate):
            return candidate
    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit("Install kagglehub first:  pip install kagglehub") from error
    return kagglehub.dataset_download(KAGGLE_DATASET)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/GenImage")
    parser.add_argument("--download", action="store_true", help=f"fetch {KAGGLE_DATASET} with kagglehub")
    args = parser.parse_args()

    root = args.data_root
    if args.download and not os.path.isdir(root):
        root = download()
        print(f"dataset available at: {root}")

    generators = discover_generators(root)
    if not generators:
        raise SystemExit(f"No generator folders with train|val/ai|nature found under '{root}'.")

    print(f"\n{'generator':<12}{'train real':>11}{'train fake':>11}{'test real':>11}{'test fake':>11}   directory")
    for name, directory in generators.items():
        counts = []
        for split in ("train", "val"):
            try:
                samples = list_images(directory, split)
            except RuntimeError:
                samples = []
            fake = sum(label for _, label in samples)
            counts += [len(samples) - fake, fake]
        print(f"{name:<12}{counts[0]:>11}{counts[1]:>11}{counts[2]:>11}{counts[3]:>11}   {directory}")
    print(f"\nDATA_ROOT={root}")


if __name__ == "__main__":
    main()
