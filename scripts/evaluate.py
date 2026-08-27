"""
LOTA Multi-Generator Evaluation CLI
Loads a trained checkpoint and evaluates it across all 8 GenImage generator test splits.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch

from src.utils.config_parser import load_config
from src.models import create_model
from src.evaluation.evaluate import evaluate_model_on_generators
from src.evaluation.robustness import evaluate_model_robustness
from src.evaluation.visualization import plot_generalization_heatmap, plot_robustness_curves
from src.data.dataset import GenImageDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate LOTA Model across Generators")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pth")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock test data")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"[CHECKPOINT] Loaded weights from {args.checkpoint}")

    # 1. Evaluate on all generators
    results = evaluate_model_on_generators(
        model=model,
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        use_mock_data=args.mock,
        device=device
    )

    # 2. Evaluate robustness
    val_ds = GenImageDataset(
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        split="val",
        use_mock_data=args.mock,
        mock_num_samples=64
    )
    robustness_res = evaluate_model_robustness(model=model, base_dataset=val_ds, device=device)

    # 3. Plots
    plot_generalization_heatmap({"Evaluation": results})
    plot_robustness_curves(robustness_res)


if __name__ == "__main__":
    main()
