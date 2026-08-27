"""
Real-Data Readiness Gate for LOTA
Verifies whether an actual GenImage dataset subset is available for real-world forensic validation and training.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from src.data.dataset import GenImageDataset


def check_genimage_readiness(root_dir: str = "./data/GenImage") -> bool:
    print("=" * 75)
    print("  LOTA REAL-DATA VALIDATION READINESS GATE")
    print("=" * 75)
    print(f"Target Dataset Root: {os.path.abspath(root_dir)}\n")

    if not os.path.exists(root_dir):
        print("  [CHECK 1] Dataset Root Directory : MISSING")
        print("  [CHECK 2] Generator Folders      : MISSING")
        print("  [CHECK 3] Real Nature Images     : MISSING")
        print("  [CHECK 4] Fake AI Images         : MISSING")
        print("  [CHECK 5] Image Loading Test     : FAILED\n")
        print("-" * 75)
        print("  REAL-DATA VALIDATION STATUS: NOT READY")
        print("-" * 75)
        print("\n[ACTION REQUIRED FOR USER]:")
        print("1. Download a minimal GenImage subset (e.g. SD v1.5 val split).")
        print("2. Place images under:")
        print("     data/GenImage/sd15/val/nature/   (Real images, e.g. 0_nature.png)")
        print("     data/GenImage/sd15/val/ai/       (AI images, e.g. 0_ai.png)")
        print("3. Re-run this check to unlock REAL-DATA forensic validation and model training.\n")
        return False

    # Check generators present
    generators = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"  [CHECK 1] Dataset Root Directory : FOUND ({root_dir})")
    print(f"  [CHECK 2] Generator Folders      : FOUND ({generators})")

    ds = GenImageDataset(root_dir=root_dir, split="val", use_mock_data=False)
    if ds.use_mock_data or len(ds.samples) < 2:
        print("  [CHECK 3] Sample Population      : INSUFFICIENT (< 2 real/fake files)")
        print("-" * 75)
        print("  REAL-DATA VALIDATION STATUS: NOT READY")
        print("-" * 75)
        return False

    # Check real and fake existence
    has_real = any(s["label"] == 0 for s in ds.samples)
    has_fake = any(s["label"] == 1 for s in ds.samples)
    print(f"  [CHECK 3] Real Nature Images     : {'FOUND' if has_real else 'MISSING'}")
    print(f"  [CHECK 4] Fake AI Images         : {'FOUND' if has_fake else 'MISSING'}")

    # Test load
    try:
        sample = ds[0]
        print(f"  [CHECK 5] Image Loading Test     : PASSED (Shape: {sample['raw_image'].shape})")
    except Exception as e:
        print(f"  [CHECK 5] Image Loading Test     : FAILED ({e})")
        print("-" * 75)
        print("  REAL-DATA VALIDATION STATUS: NOT READY")
        print("-" * 75)
        return False

    if has_real and has_fake:
        print("-" * 75)
        print("  REAL-DATA VALIDATION STATUS: READY")
        print("-" * 75)
        print(f"[READY] {len(ds.samples)} real/fake samples successfully indexed across {len(generators)} generators.")
        return True
    else:
        print("-" * 75)
        print("  REAL-DATA VALIDATION STATUS: NOT READY")
        print("-" * 75)
        return False


if __name__ == "__main__":
    check_genimage_readiness()
