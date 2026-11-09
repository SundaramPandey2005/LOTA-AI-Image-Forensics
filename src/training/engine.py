"""Training loop shared by the LOTA baseline (K = 1) and LOTA-MP (K > 1).

Hyper-parameters follow the paper / official code (Adam, lr 1e-4, poly decay,
BCE, ImageNet-pretrained ResNet-50). The only difference between the two
methods is ``model.num_patches``.
"""
import json
import os
import time
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn.functional as F
import yaml

from src.data.genimage import discover_generators, list_images, split_train_val
from src.data.loaders import build_dataset, build_loader
from src.evaluation.evaluator import autocast, collect_patch_logits, metrics_at_k
from src.models.multi_patch import MultiPatchLOTA, aggregate_logits
from src.training.losses import multi_patch_loss
from src.utils.reproducibility import set_seed


def make_logger(path: str) -> Callable[[str], None]:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return log


def build_model(config: Dict[str, Any]) -> MultiPatchLOTA:
    model_cfg = config["model"]
    return MultiPatchLOTA(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        num_patches=model_cfg["num_patches"],
        aggregation=model_cfg["aggregation"],
        input_size=config["forensic"]["image_size"],
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[MultiPatchLOTA, Dict[str, Any]]:
    """Rebuild the model stored in ``path``; returns (model, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def poly_lr(base_lr: float, epoch: int, max_epochs: int, power: float) -> float:
    return base_lr * (1 - epoch / max_epochs) ** power


def train_model(config: Dict[str, Any], out_dir: str, device: torch.device) -> Dict[str, Any]:
    """Train one model; writes config.yaml, train.log, history.json, best.pth."""
    os.makedirs(out_dir, exist_ok=True)
    log = make_logger(os.path.join(out_dir, "train.log"))
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    seed = int(config["seed"])
    set_seed(seed)
    data_cfg, train_cfg, model_cfg = config["data"], config["train"], config["model"]
    k = int(model_cfg["num_patches"])

    generators = discover_generators(data_cfg["root"])
    if data_cfg["train_generator"] not in generators:
        raise KeyError(
            f"Train generator '{data_cfg['train_generator']}' not found under '{data_cfg['root']}'. "
            f"Available: {list(generators)}"
        )
    samples = list_images(generators[data_cfg["train_generator"]], "train", data_cfg["max_train_per_class"], seed=0)
    train_samples, val_samples = split_train_val(samples, data_cfg["val_fraction"], seed=0)

    train_set = build_dataset(train_samples, config, k, deterministic=False)
    val_set = build_dataset(val_samples, config, k, deterministic=True)
    train_loader = build_loader(train_set, train_cfg["batch_size"], True, data_cfg["num_workers"], seed)
    val_loader = build_loader(val_set, config["eval"]["batch_size"], False, data_cfg["num_workers"])

    model = build_model(config).to(device)
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    log(f"=== {config['name']} | train on {data_cfg['train_generator']} | K={k} | seed={seed} | device={device}")
    log(
        f"train images: {len(train_samples)} | held-out val images: {len(val_samples)} | "
        f"bit_mode={config['forensic']['bit_mode']} | aggregation={model_cfg['aggregation']} | "
        f"instance_loss_weight={train_cfg['instance_loss_weight']}"
    )

    history = []
    best_key, best_epoch = (-1.0, float("-inf")), 0
    checkpoint_path = os.path.join(out_dir, "best.pth")
    run_start = time.time()

    for epoch in range(1, train_cfg["epochs"] + 1):
        lr = poly_lr(train_cfg["lr"], epoch - 1, train_cfg["epochs"], train_cfg["lr_power"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        epoch_start, running, seen = time.time(), 0.0, 0
        for batch in train_loader:
            patches = batch["patches"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, use_amp):
                image_logit, patch_logits = model(patches)
            loss = multi_patch_loss(
                image_logit.float(), patch_logits.float(), labels, train_cfg["instance_loss_weight"]
            )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running / max(seen, 1)

        logits, labels_np = collect_patch_logits(model, val_loader, device, use_amp)
        val = metrics_at_k(logits, labels_np, k, model_cfg["aggregation"])
        fused = aggregate_logits(torch.from_numpy(logits[:, :k]).float(), model_cfg["aggregation"])
        val_loss = F.binary_cross_entropy_with_logits(fused, torch.from_numpy(labels_np).float()).item()

        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val["acc"],
            "val_ap": val["ap"],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        key = (val["acc"], -val_loss)
        improved = key > best_key
        if improved:
            best_key, best_epoch = key, epoch
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch, "val": record},
                checkpoint_path,
            )
        log(
            f"epoch {epoch:02d}/{train_cfg['epochs']} lr={lr:.2e} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val['acc']:.2f} val_ap={val['ap']:.2f} "
            f"time={record['seconds']:.0f}s{' *' if improved else ''}"
        )

    summary = {
        "name": config["name"],
        "best_epoch": best_epoch,
        "best_val_acc": best_key[0],
        "train_seconds": time.time() - run_start,
        "num_train_images": len(train_samples),
        "num_val_images": len(val_samples),
        "history": history,
    }
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(
        f"best epoch {best_epoch} (val_acc={best_key[0]:.2f}) | "
        f"total train time {summary['train_seconds'] / 60:.1f} min | saved {checkpoint_path}"
    )
    return summary
