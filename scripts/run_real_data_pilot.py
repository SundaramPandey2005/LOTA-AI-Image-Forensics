"""
LOTA Real-Data Pilot Gate (Gate B)
Executes empirical pilot verification strictly on legitimate GenImage images (use_mock_data=False).
Measures actual throughput, compute cost, memory behavior, and logs results with source_type='experimental'.
"""
import os
import sys
import time
import yaml
from typing import Dict, Any, Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import GenImageDataset
from src.data.splits import create_stratified_split
from src.models import create_model, NoiseBasedClassifier, NoiseGuidedClassifier, RawImageClassifier
from src.experiments.database import ExperimentDatabase
from src.experiments.logger import ExperimentLogger
from src.training.metrics import compute_classification_metrics
from src.utils.reproducibility import set_seed
from scripts.check_data_readiness import check_genimage_readiness


def run_real_data_pilot(
    config_path: str = "./configs/pilot_real_data.yaml",
    db_path: str = "./experiments/results/lota_experiments.db"
) -> bool:
    print("=" * 75)
    print("  LOTA REAL-DATA EXPERIMENT RUNNER")
    print("=" * 75)

    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "experiment_name": "real_data_pilot_biggan",
            "generator": "biggan",
            "data": {"root_dir": "./data/GenImage", "use_mock_data": False, "train_val_ratio": 0.7, "max_real_samples": 100, "max_fake_samples": 100},
            "training": {"batch_size": 16, "epochs": 3, "learning_rate": 0.0001, "optimizer": "adam", "mixed_precision": True},
            "reproducibility": {"seed": 42}
        }

    dataset_root = cfg["data"].get("root_dir", "./data/GenImage")
    gen_name = cfg.get("generator", "biggan")
    exp_id = cfg.get("experiment_name", f"real_data_pilot_{gen_name}")
    use_mock = cfg["data"].get("use_mock_data", False)

    if use_mock:
        raise ValueError("[RESEARCH INTEGRITY ERROR] Real-data experiment cannot run with use_mock_data=True.")

    # 1. Dataset Readiness Check
    if not check_genimage_readiness(dataset_root):
        print("=" * 75)
        print("  REAL-DATA EXPERIMENT: NOT READY")
        print("=" * 75)
        print("  Reason: Legitimate GenImage dataset images were not found.")
        print("  Experiment has not been run.")
        print("  No experimental performance claims have been generated.")
        print("=" * 75 + "\n")
        return False

    set_seed(cfg.get("reproducibility", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg["training"].get("mixed_precision", True) and (device.type == "cuda")

    # 2. Load Real Dataset
    print("\n[DATASET LOADING] Initializing real GenImage dataset...")
    try:
        train_exists = os.path.exists(os.path.join(dataset_root, gen_name, "train"))
        val_exists = os.path.exists(os.path.join(dataset_root, gen_name, "val"))

        require_exact = cfg["data"].get("require_exact_sample_counts", False)
        req_real = cfg["data"].get("max_real_samples", None)
        req_fake = cfg["data"].get("max_fake_samples", None)

        if train_exists and val_exists:
            train_ds = GenImageDataset(
                root_dir=dataset_root,
                generators=[gen_name],
                split="train",
                max_samples_per_class=req_real,
                use_mock_data=False
            )
            val_ds = GenImageDataset(
                root_dir=dataset_root,
                generators=[gen_name],
                split="val",
                max_samples_per_class=cfg["data"].get("max_val_samples", req_real),
                use_mock_data=False
            )
            total_real_samples = sum(1 for s in train_ds.samples if s[1] == 0) + sum(1 for s in val_ds.samples if s[1] == 0)
            total_fake_samples = sum(1 for s in train_ds.samples if s[1] == 1) + sum(1 for s in val_ds.samples if s[1] == 1)
        else:
            # Handle minimal single-split dataset (e.g. biggan/val with nature and ai)
            active_split = "val" if val_exists else "train"
            base_ds = GenImageDataset(
                root_dir=dataset_root,
                generators=[gen_name],
                split=active_split,
                use_mock_data=False
            )
            total_real_samples = sum(1 for s in base_ds.samples if s[1] == 0)
            total_fake_samples = sum(1 for s in base_ds.samples if s[1] == 1)

            # Strict verification of required sample counts before proceeding to training
            if require_exact:
                err_msgs = []
                if req_real is not None and total_real_samples < req_real:
                    err_msgs.append(f"Genuine (nature) images: {total_real_samples} available, {req_real} required (max_real_samples)")
                if req_fake is not None and total_fake_samples < req_fake:
                    err_msgs.append(f"AI generated images: {total_fake_samples} available, {req_fake} required (max_fake_samples)")
                if err_msgs:
                    print("\n" + "=" * 75)
                    print("  [RESEARCH INTEGRITY ERROR] INSUFFICIENT SAMPLES FOR CONSTRAINED BASELINE")
                    print("=" * 75)
                    print(f"  Experiment '{exp_id}' requires strict minimum dataset counts (require_exact_sample_counts=True).")
                    for msg in err_msgs:
                        print(f"  - {msg}")
                    print(f"\n  Available in '{os.path.join(dataset_root, gen_name)}':")
                    print(f"    - Genuine (nature) samples : {total_real_samples}")
                    print(f"    - AI generated samples     : {total_fake_samples}")
                    print("\n  Aborting execution before training to prevent running an undersized experiment.")
                    print("=" * 75 + "\n")
                    raise RuntimeError(
                        f"Experiment '{exp_id}' aborted: Insufficient samples for required dataset constraint. "
                        f"Available: ({total_real_samples} real, {total_fake_samples} fake). "
                        f"Required: ({req_real} real, {req_fake} fake)."
                    )

            # Cap samples if max_samples specified
            raw_samples = []
            real_count, fake_count = 0, 0
            for s in base_ds.samples:
                lbl = int(s[1])
                if lbl == 0 and (req_real is None or real_count < req_real):
                    raw_samples.append({"path": s[0], "label": 0, "generator": s[2]})
                    real_count += 1
                elif lbl == 1 and (req_fake is None or fake_count < req_fake):
                    raw_samples.append({"path": s[0], "label": 1, "generator": s[2]})
                    fake_count += 1

            train_samples, val_samples = create_stratified_split(
                raw_samples,
                train_ratio=cfg["data"].get("train_val_ratio", 0.7),
                seed=cfg.get("reproducibility", {}).get("seed", 42)
            )
            train_ds = GenImageDataset(
                root_dir=dataset_root,
                generators=[gen_name],
                samples=train_samples,
                use_mock_data=False
            )
            val_ds = GenImageDataset(
                root_dir=dataset_root,
                generators=[gen_name],
                samples=val_samples,
                use_mock_data=False
            )

        # In case require_exact was enabled on dual-split folder
        if require_exact and (train_exists and val_exists):
            err_msgs = []
            if req_real is not None and total_real_samples < req_real:
                err_msgs.append(f"Genuine (nature) images: {total_real_samples} available, {req_real} required (max_real_samples)")
            if req_fake is not None and total_fake_samples < req_fake:
                err_msgs.append(f"AI generated images: {total_fake_samples} available, {req_fake} required (max_fake_samples)")
            if err_msgs:
                raise RuntimeError(
                    f"Experiment '{exp_id}' aborted: Insufficient samples for required dataset constraint. "
                    f"Available: ({total_real_samples} real, {total_fake_samples} fake). "
                    f"Required: ({req_real} real, {req_fake} fake)."
                )

    except Exception as e:
        print(f"[ERROR] Failed to load real dataset: {e}")
        print("\n  REAL-DATA EXPERIMENT: FAILED")
        return False

    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Experiment ID        : {exp_id}")
    print(f"  Generator Target     : {gen_name}")
    print(f"  Real Train Samples   : {sum(1 for s in train_ds.samples if s[1] == 0)}")
    print(f"  Fake Train Samples   : {sum(1 for s in train_ds.samples if s[1] == 1)}")
    print(f"  Real Val Samples     : {sum(1 for s in val_ds.samples if s[1] == 0)}")
    print(f"  Fake Val Samples     : {sum(1 for s in val_ds.samples if s[1] == 1)}")
    print(f"  Image Size           : 256x256 (MGPS 32x32 Patch)")
    print(f"  Batch Size           : {batch_size}")
    print(f"  Compute Device       : {device} (AMP: {amp_enabled})")

    # 3. Model Initialization (Configuration-Driven: NBC vs NGC vs RAW_ONLY)
    arch_name = str(cfg.get("model", {}).get("architecture", "nbc")).lower()
    backbone_name = str(cfg.get("model", {}).get("backbone", "resnet50"))
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    num_classes = int(cfg.get("model", {}).get("num_classes", 1))

    if arch_name == "nbc":
        model = NoiseBasedClassifier(backbone=backbone_name, pretrained=pretrained, num_classes=num_classes)
        model_id = f"M_NBC_{backbone_name.upper()}"
    elif arch_name == "ngc":
        model = NoiseGuidedClassifier(backbone=backbone_name, pretrained=pretrained, num_classes=num_classes)
        model_id = f"M_NGC_{backbone_name.upper()}"
    elif arch_name in ("raw_only", "raw", "raw_image"):
        model = RawImageClassifier(backbone=backbone_name, pretrained=pretrained, num_classes=num_classes)
        model_id = f"M_RAW_{backbone_name.upper()}"
    else:
        raise ValueError(f"Unknown model architecture: '{arch_name}'. Supported architectures are 'nbc', 'ngc', and 'raw_only'.")

    model.to(device)

    # 4. Configurable Optimizer Selection
    opt_name = str(cfg["training"].get("optimizer", "adam")).lower()
    lr = float(cfg["training"].get("learning_rate", 1e-4))
    wd = float(cfg["training"].get("weight_decay", 1e-4))

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print(f"  Architecture         : {arch_name.upper()} ({model.__class__.__name__})")
    print(f"  Backbone             : {backbone_name} (pretrained={pretrained})")
    print(f"  Model Identifier     : {model_id}")
    print(f"  Optimizer            : {opt_name.upper()} (lr={lr}, weight_decay={wd})")

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Checkpoint Infrastructure
    ckpt_dir = cfg["training"].get("checkpoint_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_id}_best.pth")
    best_val_auroc = -1.0
    best_epoch = 1
    best_metrics = {}
    epoch_history = []

    epochs = cfg["training"].get("epochs", 3)
    total_images_processed = 0
    t_start = time.time()

    print(f"\n[START TRAINING] Executing {epochs} epochs on real GenImage samples ({arch_name.upper()})...")
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            patches = batch["noise_patch"].to(device)
            raw_images = batch["raw_image"].to(device) if "raw_image" in batch else None
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if arch_name == "ngc":
                    logits = model(noise_patch=patches, raw_image=raw_images).view(-1)
                elif arch_name in ("raw_only", "raw", "raw_image"):
                    logits = model(raw_image=raw_images).view(-1)
                else:
                    logits = model(noise_patch=patches).view(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            total_images_processed += len(labels)

        train_loss /= len(train_ds)

        # Per-epoch validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                patches = batch["noise_patch"].to(device)
                raw_images = batch["raw_image"].to(device) if "raw_image" in batch else None
                labels = batch["label"].to(device).float()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if arch_name == "ngc":
                        logits = model(noise_patch=patches, raw_image=raw_images).view(-1)
                    elif arch_name in ("raw_only", "raw", "raw_image"):
                        logits = model(raw_image=raw_images).view(-1)
                    else:
                        logits = model(noise_patch=patches).view(-1)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= len(val_ds)
        t_epoch = time.time() - t_epoch_start

        epoch_metrics = compute_classification_metrics(
            all_labels,
            all_preds,
            threshold=0.5
        )
        epoch_metrics["train_loss"] = train_loss
        epoch_metrics["val_loss"] = val_loss
        epoch_history.append({"epoch": epoch, **epoch_metrics})

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ACC: {epoch_metrics['accuracy']*100:.2f}% | "
            f"Val AUROC: {epoch_metrics['auroc']:.4f} | "
            f"Val AP: {epoch_metrics['average_precision']:.4f} | "
            f"Val F1: {epoch_metrics['f1']:.4f} | "
            f"Duration: {t_epoch:.2f}s"
        )

        # Save Best Checkpoint based on Validation AUROC
        if epoch_metrics["auroc"] > best_val_auroc:
            best_val_auroc = epoch_metrics["auroc"]
            best_epoch = epoch
            best_metrics = epoch_metrics.copy()
            torch.save({
                "epoch": epoch,
                "best_epoch": epoch,
                "experiment_id": exp_id,
                "architecture": arch_name,
                "model_id": model_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": epoch_metrics,
                "validation_metrics": epoch_metrics,
                "config": cfg,
                "seed": cfg.get("reproducibility", {}).get("seed", 42)
            }, best_ckpt_path)
            print(f"    --> [BEST CHECKPOINT] Saved model to {best_ckpt_path} (Val AUROC: {best_val_auroc:.4f})")

    total_duration = time.time() - t_start
    throughput = total_images_processed / total_duration if total_duration > 0 else 0.0

    final_epoch_metrics = epoch_metrics.copy()
    final_logged_metrics = best_metrics if best_metrics else final_epoch_metrics

    print("\n" + "=" * 75)
    print("  EXPERIMENT MEASUREMENTS & SUMMARY")
    print("=" * 75)
    print(f"  Experiment ID          : {exp_id}")
    print(f"  Architecture           : {arch_name.upper()}")
    print(f"  Model ID               : {model_id}")
    print(f"  Total Images Processed : {total_images_processed}")
    print(f"  Total Duration         : {total_duration:.2f}s")
    print(f"  Training Throughput    : {throughput:.2f} images/sec")
    print(f"  Best Epoch             : {best_epoch} / {epochs}")
    print(f"  Best Val AUROC         : {final_logged_metrics['auroc']:.4f}")
    print(f"  Best Val AP            : {final_logged_metrics['average_precision']:.4f}")
    print(f"  Best Val ACC           : {final_logged_metrics['accuracy'] * 100:.2f}%")
    print(f"  Best Val Precision     : {final_logged_metrics['precision']:.4f}")
    print(f"  Best Val Recall        : {final_logged_metrics['recall']:.4f}")
    print(f"  Best Val F1            : {final_logged_metrics['f1']:.4f}")
    print(f"  Best Train Loss        : {final_logged_metrics.get('train_loss', 0.0):.4f}")
    print(f"  Best Val Loss          : {final_logged_metrics.get('val_loss', 0.0):.4f}")
    print(f"  Final Epoch AUROC      : {final_epoch_metrics['auroc']:.4f}")
    print(f"  Final Epoch ACC        : {final_epoch_metrics['accuracy'] * 100:.2f}%")

    # Enriched config payload for SQLite persistence
    resolved_cfg = dict(cfg)
    resolved_cfg["training_summary"] = {
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "best_val_metrics": final_logged_metrics,
        "final_epoch_metrics": final_epoch_metrics,
        "epoch_history": epoch_history
    }

    # Resolve human-readable experiment name
    if "sanity" in exp_id.lower() or "raw" in exp_id.lower():
        exp_display_name = f"E2 BigGAN Raw-Only Sanity Baseline"
    elif "e2" in exp_id.lower():
        exp_display_name = f"E2 BigGAN NGC Baseline"
    elif "e1" in exp_id.lower():
        exp_display_name = f"E1 BigGAN Constrained Baseline"
    else:
        exp_display_name = f"Real-Data {gen_name.upper()} {arch_name.upper()} Pilot"

    # Log to database as experimental (Recording Best Metrics under split='val_best')
    logger = ExperimentLogger(db_path)
    logger.log_run(
        experiment_id=exp_id,
        name=exp_display_name,
        config=resolved_cfg,
        model_id=model_id,
        metrics_by_generator={gen_name: final_logged_metrics},
        split="val_best",
        source_type="experimental",
        is_mock=False,
        training_time_sec=total_duration,
        notes=f"Arch: {arch_name.upper()} | Best epoch: {best_epoch}/{epochs} | Best Val AUROC: {final_logged_metrics['auroc']:.4f} | Final Val AUROC: {final_epoch_metrics['auroc']:.4f}"
    )

    # Record final epoch metrics under split='val_final' to clearly distinguish best vs final
    logger.db.insert_metrics(
        experiment_id=exp_id,
        generator_id=gen_name,
        split="val_final",
        source_type="experimental",
        is_mock=False,
        metrics=final_epoch_metrics
    )

    print("-" * 75)
    print(f"  EXPERIMENT {exp_id.upper()}: COMPLETED")
    print("-" * 75)
    print(f"  [SUCCESS] Best & final metrics successfully recorded in SQLite database.\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LOTA Real-Data Experiment")
    parser.add_argument("--config", type=str, default="./configs/pilot_real_data.yaml", help="Path to config YAML")
    parser.add_argument("--db_path", type=str, default="./experiments/results/lota_experiments.db", help="Database path")
    args = parser.parse_args()

    run_real_data_pilot(config_path=args.config, db_path=args.db_path)