"""Aggregate the outputs of run_experiments.py into the reported tables.

    python scripts/summarize_results.py --out outputs

Writes <out>/summary.md and <out>/summary.json. Every number is read from the
eval_*.json / latency.json files produced by the runs; nothing is hard-coded.

Rows of the main comparison:
    LOTA  (baseline)          checkpoint trained with K=1, evaluated with K=1
    LOTA + MP inference       the same baseline checkpoint, fused over K=4
    LOTA-MP (proposed)        checkpoint trained with K=4, evaluated with K=4
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

PROPOSED_K = 4


def load_runs(out_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for directory in sorted(glob.glob(os.path.join(out_dir, "runs", "*"))):
        name = os.path.basename(directory)
        if "_s" not in name:
            continue
        stem, seed = name.rsplit("_s", 1)
        evals = {}
        for path in glob.glob(os.path.join(directory, "eval_*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                evals[os.path.basename(path)[5:-5]] = json.load(handle)
        if not evals:
            continue
        first = next(iter(evals.values()))
        generator = first.get("train_generator", stem.rsplit("_", 1)[-1])
        method = stem[: -len(generator) - 1] if stem.endswith("_" + generator) else stem
        history = {}
        history_path = os.path.join(directory, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as handle:
                history = json.load(handle)
        runs.append({"method": method, "generator": generator, "seed": int(seed), "evals": evals,
                     "history": history})
    return runs


def mean_std(values: List[float]) -> Dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values),
            "values": [float(v) for v in values]}


def fmt(stat: Optional[Dict[str, float]], digits: int = 2) -> str:
    if stat is None:
        return "n/a"
    if stat["n"] > 1:
        return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}"


def collect(runs, method: str, generator: str, perturbation: str, k: int, aggregation: str = "logit_mean"):
    """{'average': {metric: stat}, 'per_generator': {name: {metric: stat}}, 'seeds': [...]}"""
    selected = [r for r in runs if r["method"] == method and r["generator"] == generator
                and perturbation in r["evals"] and f"K{k}" in r["evals"][perturbation]["average"]]
    if not selected:
        return None
    key = f"K{k}"
    result = {"seeds": [r["seed"] for r in selected], "average": {}, "per_generator": {}}
    for metric in ("acc", "ap", "auroc"):
        result["average"][metric] = mean_std(
            [r["evals"][perturbation]["average"][key][aggregation][metric] for r in selected])
    for name in selected[0]["evals"][perturbation]["per_generator"]:
        result["per_generator"][name] = {
            metric: mean_std([r["evals"][perturbation]["per_generator"][name][key][aggregation][metric]
                              for r in selected])
            for metric in ("acc", "ap")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    runs = load_runs(args.out)
    if not runs:
        raise SystemExit(f"No evaluated runs found under {os.path.join(args.out, 'runs')}.")
    latency = None
    if os.path.exists(os.path.join(args.out, "latency.json")):
        with open(os.path.join(args.out, "latency.json"), "r", encoding="utf-8") as handle:
            latency = json.load(handle)

    generators = sorted({r["generator"] for r in runs if r["method"] in ("lota", "lota_mp")},
                        key=lambda g: -sum(r["generator"] == g for r in runs))
    lines: List[str] = ["# Results summary", "",
                        "All numbers are percentages on the GenImage test (`val`) folders; "
                        "mean ± std over seeds where more than one seed was run.", ""]
    summary: Dict[str, Any] = {"main": {}, "k_ablation": {}, "aggregation": {}, "robustness": {},
                               "ablation": {}, "latency": latency}

    rows = (("LOTA (baseline, K=1)", "lota", 1),
            (f"LOTA + MP inference only (K={PROPOSED_K})", "lota", PROPOSED_K),
            (f"LOTA-MP (proposed, K={PROPOSED_K})", "lota_mp", PROPOSED_K))

    for generator in generators:
        base = collect(runs, "lota", generator, "clean", 1)
        if base is None:
            continue
        tests = list(base["per_generator"])
        lines += [f"## Trained on `{generator}` — clean test images (seeds: {base['seeds']})", "",
                  "| Method | " + " | ".join(tests) + " | Avg ACC | Avg AP |",
                  "|---|" + "---|" * (len(tests) + 2)]
        summary["main"][generator] = {}
        for label, method, k in rows:
            result = collect(runs, method, generator, "clean", k)
            if result is None:
                continue
            summary["main"][generator][label] = result
            lines.append(f"| {label} | " + " | ".join(fmt(result["per_generator"][t]["acc"]) for t in tests)
                         + f" | **{fmt(result['average']['acc'])}** | {fmt(result['average']['ap'])} |")
        prop = collect(runs, "lota_mp", generator, "clean", PROPOSED_K)
        if prop is not None:
            common = sorted(set(base["seeds"]) & set(prop["seeds"]))
            deltas = defaultdict(list)
            for seed in common:
                for metric in ("acc", "ap"):
                    b = base["average"][metric]["values"][base["seeds"].index(seed)]
                    p = prop["average"][metric]["values"][prop["seeds"].index(seed)]
                    deltas[metric].append(p - b)
            if common:
                summary["main"][generator]["delta"] = {m: mean_std(v) for m, v in deltas.items()}
                lines += ["", f"Δ (proposed − baseline), paired over seeds {common}: "
                              f"ACC {np.mean(deltas['acc']):+.2f} (per seed: "
                              + ", ".join(f"{d:+.2f}" for d in deltas["acc"]) + f"), AP {np.mean(deltas['ap']):+.2f}"]
        lines.append("")

        lines += [f"### Number of fused patches K (trained on `{generator}`, Avg ACC / Avg AP)", "",
                  "| Checkpoint | K=1 | K=2 | K=4 | K=8 |", "|---|---|---|---|---|"]
        summary["k_ablation"][generator] = {}
        for label, method in (("LOTA (trained K=1)", "lota"), (f"LOTA-MP (trained K={PROPOSED_K})", "lota_mp")):
            cells = []
            for k in (1, 2, 4, 8):
                result = collect(runs, method, generator, "clean", k)
                cells.append("n/a" if result is None
                             else f"{fmt(result['average']['acc'])} / {fmt(result['average']['ap'])}")
                if result is not None:
                    summary["k_ablation"][generator].setdefault(label, {})[f"K{k}"] = result["average"]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

        lines += [f"### Fusion rule at K={PROPOSED_K} (trained on `{generator}`, Avg ACC)", "",
                  "| Checkpoint | logit mean (proposed) | probability mean | majority vote |", "|---|---|---|---|"]
        summary["aggregation"][generator] = {}
        for label, method in (("LOTA (trained K=1)", "lota"), (f"LOTA-MP (trained K={PROPOSED_K})", "lota_mp")):
            cells = []
            for aggregation in ("logit_mean", "prob_mean", "vote"):
                result = collect(runs, method, generator, "clean", PROPOSED_K, aggregation)
                cells.append("n/a" if result is None else fmt(result["average"]["acc"]))
                if result is not None:
                    summary["aggregation"][generator].setdefault(label, {})[aggregation] = result["average"]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

        perturbations = sorted({p for r in runs if r["generator"] == generator for p in r["evals"]} - {"clean"})
        if perturbations:
            lines += [f"### Robustness to unseen degradations (trained on `{generator}`, Avg ACC)", "",
                      "| Method | clean | " + " | ".join(perturbations) + " |",
                      "|---|" + "---|" * (len(perturbations) + 1)]
            summary["robustness"][generator] = {}
            for label, method, k in rows:
                cells = []
                for perturbation in ["clean"] + perturbations:
                    result = collect(runs, method, generator, perturbation, k)
                    cells.append("n/a" if result is None else fmt(result["average"]["acc"]))
                    if result is not None:
                        summary["robustness"][generator].setdefault(label, {})[perturbation] = result["average"]
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")

    extra = sorted({(r["method"], r["generator"]) for r in runs if r["method"] not in ("lota", "lota_mp")})
    if extra:
        lines += ["## Additional ablations (clean test images, Avg ACC / Avg AP)", "",
                  "| Run | Trained on | Evaluated with | Avg ACC | Avg AP |", "|---|---|---|---|---|"]
        for method, generator in extra:
            k = 1 if "resize" in method else PROPOSED_K
            result = collect(runs, method, generator, "clean", k)
            if result is None:
                continue
            summary["ablation"][f"{method}:{generator}"] = result
            lines.append(f"| `{method}` | {generator} | K={k} | {fmt(result['average']['acc'])} | "
                         f"{fmt(result['average']['ap'])} |")
        lines.append("")

    if latency:
        lines += ["## Efficiency", "",
                  f"Device: {latency['device']}; batch size 1; median over {latency['num_images']} images. "
                  f"Parameters: {latency['params_million']:.2f} M for every K (shared backbone).", "",
                  "| K | patch extraction (ms) | classification (ms) | total (ms) |", "|---|---|---|---|"]
        for key, value in latency["classification_ms"].items():
            lines.append(f"| {key[1:]} | {latency['extraction_ms']:.2f} | {value:.2f} | "
                         f"{latency['total_ms'][key]:.2f} |")
        lines.append("")

    times = defaultdict(list)
    for run in runs:
        if run["history"]:
            times[run["method"]].append(run["history"]["train_seconds"] / 60)
    if times:
        lines += ["## Training time", "", "| Run | minutes per training run (mean) | runs |", "|---|---|---|"]
        lines += [f"| `{method}` | {np.mean(values):.1f} | {len(values)} |" for method, values in sorted(times.items())]
        lines.append("")
        summary["train_minutes"] = {method: float(np.mean(values)) for method, values in times.items()}

    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:  # consoles without UTF-8 (the files are still UTF-8)
        print(text.encode("ascii", "replace").decode("ascii"))
    print(f"\nwritten: {os.path.join(args.out, 'summary.md')} and summary.json")


if __name__ == "__main__":
    main()
