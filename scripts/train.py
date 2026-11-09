"""Train the LOTA baseline or the proposed LOTA-MP.

    python scripts/train.py --config configs/baseline_lota.yaml --data-root data/GenImage
    python scripts/train.py --config configs/proposed_lota_mp.yaml --data-root data/GenImage

Any config value can be overridden, e.g. ``--set data.train_generator=biggan train.epochs=5``.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.engine import train_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config (configs/*.yaml)")
    parser.add_argument("--data-root", default=None, help="GenImage root (overrides data.root)")
    parser.add_argument("--out", default=None, help="output directory (default: outputs/runs/<name>_<gen>_s<seed>)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.seed is not None:
        config["seed"] = args.seed

    out_dir = args.out or os.path.join(
        "outputs", "runs", f"{config['name']}_{config['data']['train_generator']}_s{config['seed']}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(config, out_dir, device)


if __name__ == "__main__":
    main()
