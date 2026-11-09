"""Run the full baseline-vs-proposed experiment matrix (resumable).

    python scripts/run_experiments.py --data-root data/GenImage --out outputs

Phases, in priority order (finished jobs are skipped when the script is re-run):
    core      train LOTA (K=1) and LOTA-MP (K=4) on the primary generator for
              every seed and evaluate both on all generators
    latency   parameter count and per-image latency of both decision rules
    second    the same comparison for the remaining training generators
    robust    evaluation under unseen JPEG compression / Gaussian blur
    ablation  LOTA-MP with instance supervision; baseline with the earlier
              resize-before-bit-extraction ordering

``--quick`` runs every phase on a tiny subset for one epoch (smoke test).
"""
import argparse
import copy
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.genimage import discover_generators, list_images  # noqa: E402
from src.evaluation.evaluator import benchmark_latency, evaluate_generators  # noqa: E402
from src.training.engine import load_checkpoint, make_logger, train_model  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(REPO, "configs", "baseline_lota.yaml")
PROPOSED = os.path.join(REPO, "configs", "proposed_lota_mp.yaml")
PHASES = ["core", "latency", "second", "robust", "ablation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--train-generators", nargs="+", default=["sdv5", "biggan"],
                        help="first one is the primary training generator")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--second-seeds", nargs="+", type=int, default=[0],
                        help="seeds for the non-primary training generators")
    parser.add_argument("--perturbations", nargs="*", default=["jpeg90", "blur1"])
    parser.add_argument("--phases", nargs="+", default=PHASES, choices=PHASES)
    parser.add_argument("--time-budget-hours", type=float, default=None,
                        help="do not start new jobs after this many hours")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test of every phase")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = os.path.join(args.out, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    log = make_logger(os.path.join(args.out, "run_experiments.log"))
    start = time.time()

    overrides = list(args.set)
    if args.quick:
        overrides = [
            "data.max_train_per_class=48", "data.max_test_per_class=16", "train.epochs=1",
            "train.batch_size=8", "eval.batch_size=8",
        ] + overrides
        args.seeds, args.second_seeds = args.seeds[:1], args.second_seeds[:1]

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(args.out, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)
    log(f"environment: {json.dumps(environment)}")

    generators = discover_generators(args.data_root)
    train_generators = [g for g in args.train_generators if g in generators]
    missing = [g for g in args.train_generators if g not in generators]
    log(f"device={device} | generators found: {list(generators)}")
    if missing:
        log(f"[warn] requested training generators not found and skipped: {missing}")
    if not train_generators:
        raise SystemExit("None of the requested training generators exist under --data-root.")
    primary = train_generators[0]

    def out_of_time() -> bool:
        return args.time_budget_hours is not None and (time.time() - start) / 3600 > args.time_budget_hours

    def make_config(base: str, generator: str, seed: int, extra=()):
        config = load_config(base, overrides + list(extra))
        config["data"]["root"] = args.data_root
        config["data"]["train_generator"] = generator
        config["seed"] = seed
        return config

    def run_dir(config) -> str:
        return os.path.join(runs_dir, f"{config['name']}_{config['data']['train_generator']}_s{config['seed']}")

    def guarded(description: str, done_file: str, job) -> None:
        if os.path.exists(done_file):
            log(f"[skip] {description} (found {os.path.relpath(done_file, args.out)})")
            return
        if out_of_time():
            log(f"[time budget reached] not starting: {description}")
            return
        log(f"[start] {description}")
        try:
            job()
        except Exception:  # keep going: one failed job must not waste the whole session
            log(f"[FAILED] {description}\n{traceback.format_exc()}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def train(config) -> None:
        directory = run_dir(config)
        guarded(f"train {os.path.basename(directory)}", os.path.join(directory, "history.json"),
                lambda: train_model(copy.deepcopy(config), directory, device))

    def evaluate(config, perturbation: str = "clean") -> None:
        directory = run_dir(config)
        target = os.path.join(directory, f"eval_{perturbation}.json")

        def job() -> None:
            model, checkpoint = load_checkpoint(os.path.join(directory, "best.pth"), device)
            eval_config = apply_overrides(checkpoint["config"], overrides)
            eval_config["data"]["root"] = args.data_root
            results, raw = evaluate_generators(
                model, eval_config, generators, device, None if perturbation == "clean" else perturbation, log
            )
            results["train_generator"] = eval_config["data"]["train_generator"]
            results["trained_num_patches"] = eval_config["model"]["num_patches"]
            results["seed"] = eval_config["seed"]
            np.savez_compressed(os.path.join(directory, f"logits_{perturbation}.npz"), **raw)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)

        if os.path.exists(os.path.join(directory, "best.pth")) or os.path.exists(target):
            guarded(f"eval {os.path.basename(directory)} [{perturbation}]", target, job)
        else:
            log(f"[skip] eval {os.path.basename(directory)} [{perturbation}]: no checkpoint")

    def comparison(generator: str, seeds) -> None:
        for seed in seeds:
            for base in (BASELINE, PROPOSED):
                config = make_config(base, generator, seed)
                train(config)
                evaluate(config)

    def latency() -> None:
        target = os.path.join(args.out, "latency.json")

        def job() -> None:
            directory = run_dir(make_config(BASELINE, primary, args.seeds[0]))
            model, checkpoint = load_checkpoint(os.path.join(directory, "best.pth"), device)
            samples = list_images(generators[primary], "val", 64, seed=0)
            result = benchmark_latency(model, checkpoint["config"], samples, device)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            log(f"  latency: {json.dumps(result)}")

        guarded("latency benchmark", target, job)

    for phase in args.phases:
        log(f"\n######## phase: {phase} | elapsed {(time.time() - start) / 60:.1f} min ########")
        if phase == "core":
            comparison(primary, args.seeds)
        elif phase == "latency":
            latency()
        elif phase == "second":
            for generator in train_generators[1:]:
                comparison(generator, args.second_seeds)
        elif phase == "robust":
            for seed in args.seeds:
                for base in (BASELINE, PROPOSED):
                    for perturbation in args.perturbations:
                        evaluate(make_config(base, primary, seed), perturbation)
        elif phase == "ablation":
            seed = args.seeds[0]
            for name, base, extra in (
                ("lota_mp_inst", PROPOSED, ["train.instance_loss_weight=1.0"]),
                ("lota_resize_first", BASELINE, ["forensic.resize_before_bits=true"]),
            ):
                config = make_config(base, primary, seed, extra)
                config["name"] = name
                train(config)
                evaluate(config)

    log(f"\nall phases finished in {(time.time() - start) / 60:.1f} min. "
        f"Next: python scripts/summarize_results.py --out {args.out}")


if __name__ == "__main__":
    main()
