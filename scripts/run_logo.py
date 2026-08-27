"""
LOTA LOGO Automated CLI Runner
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from scripts.run_logo_experiments import run_logo_rotation, LOGO_REPRESENTATIVE_GENERATORS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOTA LOGO Experiment")
    parser.add_argument("--excluded_generator", type=str, default="midjourney", choices=LOGO_REPRESENTATIVE_GENERATORS)
    parser.add_argument("--config", type=str, default="./configs/logo_matrix.yaml")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for verification")
    args = parser.parse_args()

    run_logo_rotation(args.excluded_generator, args.config, use_mock_data=args.mock)
