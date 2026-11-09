"""Cross-generator evaluation.

Every test image is reduced once to its ``max_patches`` best low-bit patches
and the per-patch logits are stored. Any K <= max_patches and any fusion rule
can then be evaluated from the same forward pass, on identical patches:
K = 1 is the original LOTA decision, K > 1 is LOTA-MP.
"""
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from src.data.genimage import list_images
from src.data.loaders import build_dataset, build_loader
from src.models.multi_patch import AGGREGATIONS, MultiPatchLOTA, aggregate_logits
from src.training.metrics import compute_classification_metrics


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def collect_patch_logits(
    model: MultiPatchLOTA,
    loader,
    device: torch.device,
    amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns per-patch logits (N, K) and labels (N,) for a whole loader."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=True)
        with autocast(device, amp):
            out = model.patch_logits(patches)
        logits.append(out.float().cpu())
        labels.append(batch["label"])
    return torch.cat(logits).numpy(), torch.cat(labels).numpy().astype(np.int64)


def metrics_at_k(logits: np.ndarray, labels: np.ndarray, k: int, aggregation: str = "logit_mean") -> Dict[str, float]:
    """ACC (threshold 0.5), AP and AUROC in percent when fusing the first ``k`` patches."""
    fused = aggregate_logits(torch.from_numpy(logits[:, :k]).float(), aggregation)
    prob = torch.sigmoid(fused).numpy()
    m = compute_classification_metrics(labels, prob)
    return {"acc": 100.0 * m["accuracy"], "ap": 100.0 * m["ap"], "auroc": 100.0 * m["auroc"]}


def summarise_logits(logits: np.ndarray, labels: np.ndarray, k_values: Iterable[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in k_values:
        if k > logits.shape[1]:
            continue
        out[f"K{k}"] = {agg: metrics_at_k(logits, labels, k, agg) for agg in AGGREGATIONS}
    return out


def evaluate_generators(
    model: MultiPatchLOTA,
    config: Dict[str, Any],
    generators: Dict[str, str],
    device: torch.device,
    perturbation: Optional[str] = None,
    log=print,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Evaluate on the ``val`` split (the GenImage test set) of every generator.

    Returns (results, raw) where ``raw`` holds logits/labels per generator.
    """
    eval_cfg, data_cfg = config["eval"], config["data"]
    k_values: List[int] = list(eval_cfg["k_values"])
    tag = perturbation or "clean"
    results: Dict[str, Any] = {"perturbation": tag, "per_generator": {}, "average": {}}
    raw: Dict[str, np.ndarray] = {}

    for name, directory in generators.items():
        samples = list_images(directory, "val", data_cfg["max_test_per_class"], seed=0)
        dataset = build_dataset(samples, config, eval_cfg["max_patches"], deterministic=True, perturbation=perturbation)
        loader = build_loader(dataset, eval_cfg["batch_size"], False, data_cfg["num_workers"])
        start = time.time()
        logits, labels = collect_patch_logits(model, loader, device, config["train"]["amp"])
        entry = summarise_logits(logits, labels, k_values)
        shown = "  ".join(f"{key}={value['logit_mean']['acc']:.2f}" for key, value in entry.items())
        entry["num_images"] = int(len(labels))
        entry["num_fake"] = int(labels.sum())
        results["per_generator"][name] = entry
        raw[f"{name}_logits"], raw[f"{name}_labels"] = logits, labels
        log(f"  [{tag}] {name:<11} n={len(labels):<5} ACC: {shown}  ({time.time() - start:.0f}s)")

    names = list(results["per_generator"])
    for k in k_values:
        key = f"K{k}"
        if not all(key in results["per_generator"][n] for n in names):
            continue
        results["average"][key] = {
            agg: {
                metric: float(np.mean([results["per_generator"][n][key][agg][metric] for n in names]))
                for metric in ("acc", "ap", "auroc")
            }
            for agg in AGGREGATIONS
        }
    shown = "  ".join(f"{key}={value['logit_mean']['acc']:.2f}" for key, value in results["average"].items())
    log(f"  [{tag}] AVERAGE ACC over {len(names)} generators: {shown}")
    return results, raw


@torch.no_grad()
def benchmark_latency(
    model: MultiPatchLOTA,
    config: Dict[str, Any],
    samples: List[Tuple[str, int]],
    device: torch.device,
    num_images: int = 100,
    warmup: int = 10,
) -> Dict[str, Any]:
    """Per-image latency (batch size 1), split as in Table 8 of the paper.

    ``extraction_ms``: low-bit image + candidate scoring + top-K selection
    (CPU, image already decoded). ``classification_ms[K]``: forward pass of K
    patches through the shared backbone. Medians over ``num_images`` images.
    """
    model.eval()
    eval_cfg = config["eval"]
    dataset = build_dataset(samples[: num_images + warmup], config, eval_cfg["max_patches"], deterministic=True)
    images = [dataset._load(path) for path, _ in dataset.samples]
    warmup = min(warmup, len(images) // 2)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    extraction: List[float] = []
    classification: Dict[int, List[float]] = {k: [] for k in eval_cfg["k_values"] if k <= eval_cfg["max_patches"]}
    for i, (image, (path, _)) in enumerate(zip(images, dataset.samples)):
        start = time.perf_counter()
        patches, _ = dataset.extract(image, dataset._rng(path))
        elapsed = (time.perf_counter() - start) * 1e3
        tensor = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        if i >= warmup:
            extraction.append(elapsed)
        for k in classification:
            sync()
            start = time.perf_counter()
            with autocast(device, config["train"]["amp"]):
                model.patch_logits(tensor[:, :k]).mean(dim=1)
            sync()
            if i >= warmup:
                classification[k].append((time.perf_counter() - start) * 1e3)

    extraction_ms = float(np.median(extraction))
    result: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "num_images": len(extraction),
        "extraction_ms": extraction_ms,
        "classification_ms": {f"K{k}": float(np.median(v)) for k, v in classification.items()},
    }
    result["total_ms"] = {k: extraction_ms + v for k, v in result["classification_ms"].items()}
    result["params_million"] = sum(p.numel() for p in model.parameters()) / 1e6
    return result
