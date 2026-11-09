"""Evaluate a trained checkpoint on the test split of every GenImage generator.

    python scripts/evaluate.py --checkpoint outputs/runs/lota_sdv5_s0/best.pth --data-root data/GenImage

Reports ACC / AP for K = 1, 2, 4, 8 fused patches from one forward pass
(K = 1 is the original LOTA decision rule). Optional:
    --perturbation jpeg90 blur1     robustness to unseen degradations (Fig. 6)
    --latency                       per-image latency and parameter count
Results are written next to the checkpoint as eval_<perturbation>.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.genimage import discover_generators, list_images  # noqa: E402
from src.evaluation.evaluator import benchmark_latency, evaluate_generators  # noqa: E402
from src.training.engine import load_checkpoint, make_logger  # noqa: E402
from src.utils.config import apply_overrides  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None, help="GenImage root (default: the one stored in the checkpoint)")
    parser.add_argument("--out", default=None, help="output directory (default: the checkpoint's directory)")
    parser.add_argument("--generators", nargs="*", default=None, help="subset of test generators (default: all found)")
    parser.add_argument("--perturbation", nargs="*", default=["clean"], help="clean, jpeg<quality>, blur<sigma>")
    parser.add_argument("--latency", action="store_true", help="also benchmark latency")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = apply_overrides(checkpoint["config"], args.set)
    if args.data_root:
        config["data"]["root"] = args.data_root
    out_dir = args.out or os.path.dirname(os.path.abspath(args.checkpoint))
    log = make_logger(os.path.join(out_dir, "eval.log"))

    generators = discover_generators(config["data"]["root"])
    if args.generators:
        generators = {name: generators[name] for name in args.generators}
    log(f"=== evaluating {args.checkpoint} (epoch {checkpoint['epoch']}) on {list(generators)}")

    for perturbation in args.perturbation:
        results, raw = evaluate_generators(
            model, config, generators, device, None if perturbation == "clean" else perturbation, log
        )
        results["checkpoint"] = args.checkpoint
        results["train_generator"] = config["data"]["train_generator"]
        results["trained_num_patches"] = config["model"]["num_patches"]
        with open(os.path.join(out_dir, f"eval_{perturbation}.json"), "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        np.savez_compressed(os.path.join(out_dir, f"logits_{perturbation}.npz"), **raw)

    if args.latency:
        samples = list_images(next(iter(generators.values())), "val", 64, seed=0)
        latency = benchmark_latency(model, config, samples, device)
        with open(os.path.join(out_dir, "latency.json"), "w", encoding="utf-8") as handle:
            json.dump(latency, handle, indent=2)
        log(f"  latency: {json.dumps(latency)}")


if __name__ == "__main__":
    main()
