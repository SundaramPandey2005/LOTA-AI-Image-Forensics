"""
LOTA Zero-Shot Cross-Generator Evaluation Script
Performs cross-generator zero-shot evaluation between:
  1. E1 (Trained on BigGAN) -> Evaluated on VQDM validation data
  2. E3 (Trained on VQDM)   -> Evaluated on BigGAN validation data
"""
import os
import sys
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_parser import load_config
from src.models import create_model
from src.data.dataset import GenImageDataset
from src.training.metrics import compute_classification_metrics


def load_model_from_checkpoint(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Instantiates the model architecture from config and loads trained weights from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: '{os.path.abspath(checkpoint_path)}'")

    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint structure in '{checkpoint_path}'")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    arch_name: str = "nbc"
) -> Dict[str, float]:
    """
    Runs model inference over the dataloader and computes classification metrics.
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        patches = batch["noise_patch"].to(device) if "noise_patch" in batch else None
        raw_images = batch.get("raw_image", None)
        if raw_images is not None:
            raw_images = raw_images.to(device)
        labels = batch["label"].to(device).float()

        if arch_name == "ngc":
            logits = model(noise_patch=patches, raw_image=raw_images).view(-1)
        elif arch_name in ("raw_only", "raw", "raw_image"):
            logits = model(raw_image=raw_images).view(-1)
        else:
            logits = model(noise_patch=patches).view(-1)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    metrics = compute_classification_metrics(all_labels, all_preds, threshold=0.5)
    return metrics


def evaluate_cross_generator_pair(
    source_name: str,
    target_generator: str,
    config_path: str,
    checkpoint_path: str,
    data_root: str = "./data/GenImage",
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Loads a model trained on source generator and evaluates zero-shot on target generator validation data.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)

    # 1. Load trained model
    model = load_model_from_checkpoint(cfg, checkpoint_path, dev)
    arch_name = cfg.get("model", {}).get("architecture", "nbc").lower()

    # 2. Build target evaluation dataset using identical preprocessing from model config
    data_cfg = cfg.get("data", {})
    image_size = data_cfg.get("image_size", 256)
    patch_size = data_cfg.get("patch_size", 32)
    bit_planes = data_cfg.get("bit_planes", [0, 1, 2])
    normalization = data_cfg.get("normalization", "thresholding")
    pre_resize_size = data_cfg.get("pre_resize_size", None)
    jpeg_reencode = data_cfg.get("jpeg_reencode_quality", data_cfg.get("jpeg_quality", None))
    max_val_samples = data_cfg.get("max_val_samples", None)

    eval_ds = GenImageDataset(
        root_dir=data_root,
        generators=[target_generator],
        split="val",
        image_size=image_size,
        patch_size=patch_size,
        bit_planes=bit_planes,
        normalization_method=normalization,
        pre_resize_size=pre_resize_size,
        jpeg_reencode_quality=jpeg_reencode,
        max_samples_per_class=max_val_samples,
        use_mock_data=use_mock,
        mock_num_samples=32 if use_mock else 64
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    real_count = sum(1 for s in eval_ds.samples if s[1] == 0)
    fake_count = sum(1 for s in eval_ds.samples if s[1] == 1)

    # 3. Compute metrics
    metrics = evaluate_on_dataset(
        model=model,
        dataloader=eval_loader,
        device=dev,
        arch_name=arch_name
    )

    result_payload = {
        "source_experiment": source_name,
        "trained_generator": cfg.get("generator", "unknown"),
        "target_generator": target_generator,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "architecture": arch_name,
        "total_samples": len(eval_ds),
        "real_samples": real_count,
        "fake_samples": fake_count,
        "metrics": metrics
    }

    return result_payload


def print_result_block(title: str, res: Dict[str, Any]):
    """
    Nicely formats and prints evaluation result block.
    """
    m = res["metrics"]
    print("=" * 80)
    print(f"  ZERO-SHOT CROSS-GENERATOR EVALUATION: {title}")
    print("=" * 80)
    print(f"  Source Model (Trained On) : {res['trained_generator'].upper()} ({res['source_experiment']})")
    print(f"  Target Evaluation Split   : {res['target_generator'].upper()} (Validation Split)")
    print(f"  Architecture Backbone     : {res['architecture'].upper()} (ResNet-50)")
    print(f"  Checkpoint File           : {res['checkpoint_path']}")
    print(f"  Total Samples Evaluated   : {res['total_samples']} ({res['real_samples']} Real, {res['fake_samples']} Fake)")
    print("-" * 80)
    print(f"  Accuracy                  : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
    print(f"  AUROC                     : {m['auroc']:.4f}")
    print(f"  Average Precision (AP)    : {m['average_precision']:.4f} ({m['average_precision']*100:.2f}%)")
    print(f"  F1 Score                  : {m['f1']:.4f}")
    print(f"  Precision                 : {m.get('precision', 0.0):.4f}")
    print(f"  Recall                    : {m.get('recall', 0.0):.4f}")
    print("=" * 80 + "\n")


def run_cross_generator_evaluations(
    e1_config: str = "./configs/biggan_constrained_baseline_e1.yaml",
    e1_checkpoint: str = "./checkpoints/biggan_constrained_baseline_e1_best.pth",
    e3_config: str = "./configs/vqdm_e3_baseline.yaml",
    e3_checkpoint: str = "./checkpoints/vqdm_e3_baseline_best.pth",
    data_root: str = "./data/GenImage",
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    use_mock: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Executes both cross-generator zero-shot evaluations:
      1. E1 BigGAN -> VQDM
      2. E3 VQDM -> BigGAN
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EVALUATION START] Initializing Zero-Shot Cross-Generator Evaluations on device: {dev}")

    results = {}

    # Evaluation 1: E1 BigGAN -> VQDM
    print("\n[RUNNING 1/2] Evaluating E1 (BigGAN-trained) model on VQDM validation data...")
    res_e1_to_vqdm = evaluate_cross_generator_pair(
        source_name="E1 BigGAN Baseline",
        target_generator="vqdm",
        config_path=e1_config,
        checkpoint_path=e1_checkpoint,
        data_root=data_root,
        batch_size=batch_size,
        device=dev,
        use_mock=use_mock
    )
    print_result_block("E1 BigGAN -> VQDM", res_e1_to_vqdm)
    results["e1_to_vqdm"] = res_e1_to_vqdm

    # Evaluation 2: E3 VQDM -> BigGAN
    print("[RUNNING 2/2] Evaluating E3 (VQDM-trained) model on BigGAN validation data...")
    res_e3_to_biggan = evaluate_cross_generator_pair(
        source_name="E3 VQDM Baseline",
        target_generator="biggan",
        config_path=e3_config,
        checkpoint_path=e3_checkpoint,
        data_root=data_root,
        batch_size=batch_size,
        device=dev,
        use_mock=use_mock
    )
    print_result_block("E3 VQDM -> BigGAN", res_e3_to_biggan)
    results["e3_to_biggan"] = res_e3_to_biggan

    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Generator Evaluation (E1 BigGAN <-> E3 VQDM)")
    parser.add_argument("--e1-config", type=str, default="./configs/biggan_constrained_baseline_e1.yaml", help="Path to E1 config YAML")
    parser.add_argument("--e1-checkpoint", type=str, default="./checkpoints/biggan_constrained_baseline_e1_best.pth", help="Path to E1 checkpoint .pth")
    parser.add_argument("--e3-config", type=str, default="./configs/vqdm_e3_baseline.yaml", help="Path to E3 config YAML")
    parser.add_argument("--e3-checkpoint", type=str, default="./checkpoints/vqdm_e3_baseline_best.pth", help="Path to E3 checkpoint .pth")
    parser.add_argument("--data-root", type=str, default="./data/GenImage", help="Root directory of GenImage dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for test/CI validation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cross_generator_evaluations(
        e1_config=args.e1_config,
        e1_checkpoint=args.e1_checkpoint,
        e3_config=args.e3_config,
        e3_checkpoint=args.e3_checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=device,
        use_mock=args.mock
    )


if __name__ == "__main__":
    main()
