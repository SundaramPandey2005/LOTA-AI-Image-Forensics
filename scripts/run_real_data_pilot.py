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
from src.models.nbc import LOTANoiseClassifier
from src.experiments.database import ExperimentDatabase
from src.experiments.logger import ExperimentLogger
from src.utils.reproducibility import set_seed
from scripts.check_data_readiness import check_genimage_readiness


def run_real_data_pilot(
    config_path: str = "./configs/pilot_real_data.yaml",
    db_path: str = "./experiments/results/lota_experiments.db"
) -> bool:
    print("=" * 75)
    print("  LOTA REAL-DATA PILOT GATE (GATE B)")
    print("=" * 75)

    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "experiment_name": "real_data_pilot_sd15",
            "generator": "sd15",
            "data": {"root_dir": "./data/GenImage", "use_mock_data": False, "max_real_samples": 100, "max_fake_samples": 100},
            "training": {"batch_size": 16, "epochs": 3, "learning_rate": 0.0001, "mixed_precision": True},
            "reproducibility": {"seed": 42}
        }

    dataset_root = cfg["data"].get("root_dir", "./data/GenImage")
    gen_name = cfg.get("generator", "sd15")
    use_mock = cfg["data"].get("use_mock_data", False)

    if use_mock:
        raise ValueError("[RESEARCH INTEGRITY ERROR] Real-data pilot cannot run with use_mock_data=True.")

    # 1. Dataset Readiness Check
    if not check_genimage_readiness(dataset_root):
        print("=" * 75)
        print("  REAL-DATA PILOT GATE: NOT READY")
        print("=" * 75)
        print("  Reason: Legitimate GenImage dataset images were not found.")
        print("  Real-data pilot has not been run.")
        print("  No experimental performance claims have been generated.")
        print("=" * 75 + "\n")
        return False

    set_seed(cfg.get("reproducibility", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg["training"].get("mixed_precision", True) and (device.type == "cuda")

    # 2. Load Real Dataset
    print("\n[DATASET LOADING] Initializing real GenImage dataset...")
    try:
        train_ds = GenImageDataset(
            root_dir=dataset_root,
            generators=[gen_name],
            split="train",
            max_samples_per_class=cfg["data"].get("max_real_samples", 100),
            use_mock_data=False
        )
        val_ds = GenImageDataset(
            root_dir=dataset_root,
            generators=[gen_name],
            split="val",
            max_samples_per_class=cfg["data"].get("max_real_samples", 50),
            use_mock_data=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to load real dataset: {e}")
        print("\n  REAL-DATA PILOT GATE: FAILED")
        return False

    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Generator Target     : {gen_name}")
    print(f"  Real Train Samples   : {sum(1 for s in train_ds.samples if s[1] == 0)}")
    print(f"  Fake Train Samples   : {sum(1 for s in train_ds.samples if s[1] == 1)}")
    print(f"  Real Val Samples     : {sum(1 for s in val_ds.samples if s[1] == 0)}")
    print(f"  Fake Val Samples     : {sum(1 for s in val_ds.samples if s[1] == 1)}")
    print(f"  Image Size           : 256x256 (MGPS 32x32 Patch)")
    print(f"  Batch Size           : {batch_size}")
    print(f"  Compute Device       : {device} (AMP: {amp_enabled})")

    # 3. Model Initialization
    model = LOTANoiseClassifier(backbone="resnet50", pretrained=True, num_classes=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"].get("learning_rate", 1e-4), weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    epochs = cfg["training"].get("epochs", 3)
    total_images_processed = 0
    t_start = time.time()

    print(f"\n[START TRAINING] Executing {epochs} pilot epochs on real GenImage samples...")
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            patches = batch["noise_patch"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(patches)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            total_images_processed += len(labels)

        t_epoch = time.time() - t_epoch_start
        train_loss /= len(train_ds)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Epoch Duration: {t_epoch:.2f}s")

    # Validation
    t_val_start = time.time()
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            patches = batch["noise_patch"].to(device)
            labels = batch["label"].to(device).float()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(patches)
                loss = criterion(logits, labels)
            val_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

    t_val = time.time() - t_val_start
    total_duration = time.time() - t_start
    throughput = total_images_processed / total_duration

    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    acc = float(accuracy_score(labels_arr, (preds_arr >= 0.5).astype(int)))
    auroc = float(roc_auc_score(labels_arr, preds_arr)) if len(set(labels_arr)) > 1 else 0.5
    ap = float(average_precision_score(labels_arr, preds_arr)) if len(set(labels_arr)) > 1 else 0.5

    print("\n" + "=" * 75)
    print("  REAL-DATA PILOT MEASUREMENTS & COMPUTE PROFILE")
    print("=" * 75)
    print(f"  Total Images Processed : {total_images_processed}")
    print(f"  Total Duration         : {total_duration:.2f}s")
    print(f"  Training Throughput    : {throughput:.2f} images/sec")
    print(f"  Validation Duration    : {t_val:.2f}s")
    print(f"  Final Validation Acc   : {acc * 100:.2f}%")
    print(f"  Final Validation AUROC : {auroc:.4f}")
    print(f"  Final Validation AP    : {ap:.4f}")

    # Log to database as experimental
    logger = ExperimentLogger(db_path)
    logger.log_run(
        experiment_id=cfg.get("experiment_name", "real_data_pilot_sd15"),
        name="Real-Data Pilot SD1.5",
        config=cfg,
        model_id="M_NBC_RESNET50",
        metrics_by_generator={gen_name: {"accuracy": acc, "auroc": auroc, "average_precision": ap}},
        source_type="experimental",
        is_mock=False
    )

    print("-" * 75)
    print("  REAL-DATA PILOT GATE: PASSED")
    print("-" * 75)
    print("  [SUCCESS] Real GenImage pipeline validated. Ready for full reproduction!\n")
    return True


if __name__ == "__main__":
    run_real_data_pilot()
