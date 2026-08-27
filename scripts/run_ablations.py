import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_parser import load_config


def main():
    print("=================================================================")
    print("            LOTA FOCUSED ABLATION STUDY RUNNER                   ")
    print("=================================================================")
    ablations_cfg = load_config("configs/ablations.yaml")
    
    print("1. Bit-Plane Depth Ablation (K = 0..5):")
    for v in ablations_cfg.get("ablation_1_bit_planes", {}).get("variants", []):
        print(f"   - {v['name']}: {v['bit_planes']}")

    print("\n2. MGPS Patch Strategy & Size Ablation:")
    print(f"   - Strategies: {ablations_cfg.get('ablation_2_mgps', {}).get('strategies', [])}")
    print(f"   - Patch Sizes: {ablations_cfg.get('ablation_2_mgps', {}).get('patch_sizes', [])}")

    print("\n3. Classifier Architecture Ablation:")
    print(f"   - Architectures: {ablations_cfg.get('ablation_3_classifiers', {}).get('architectures', [])}")


if __name__ == "__main__":
    main()
