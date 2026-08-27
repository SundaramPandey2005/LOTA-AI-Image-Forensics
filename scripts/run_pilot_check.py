"""
LOTA Unified Pilot & Readiness Gate Runner
Executes Gate A (Infrastructure Gate) and Gate B (Real-Data Pilot Gate) with strict separation.
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_infrastructure_check import run_infrastructure_gate
from scripts.run_real_data_pilot import run_real_data_pilot


def main():
    parser = argparse.ArgumentParser(description="LOTA Verification & Pilot Gates")
    parser.add_argument("--gate", choices=["infra", "real", "both"], default="both", help="Gate to run")
    parser.add_argument("--config", type=str, default="./configs/pilot_real_data.yaml", help="Pilot config path")
    args = parser.parse_args()

    infra_passed = False
    real_passed = False

    if args.gate in ["infra", "both"]:
        infra_passed = run_infrastructure_gate()

    if args.gate in ["real", "both"]:
        real_passed = run_real_data_pilot(config_path=args.config)

    print("\n" + "=" * 75)
    print("  LOTA GATE EXECUTION SUMMARY")
    print("=" * 75)
    if args.gate in ["infra", "both"]:
        print(f"  INFRASTRUCTURE GATE (Gate A) : {'PASSED' if infra_passed else 'FAILED'}")
    if args.gate in ["real", "both"]:
        print(f"  REAL-DATA PILOT GATE (Gate B): {'PASSED' if real_passed else 'NOT READY / FAILED'}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
