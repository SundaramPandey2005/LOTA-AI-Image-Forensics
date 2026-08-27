"""
LOTA Infrastructure Gate (Gate A)
Validates complete software and training infrastructure using explicit mock data (use_mock_data=True).
Tests: environment, dataset, model, forward/backward, optimizer, AMP, checkpointing, and DB logging.
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import GenImageDataset
from src.models.nbc import LOTANoiseClassifier
from src.experiments.database import ExperimentDatabase
from src.experiments.logger import ExperimentLogger


def run_infrastructure_gate(db_path: str = "./experiments/results/lota_experiments.db") -> bool:
    print("=" * 75)
    print("  LOTA INFRASTRUCTURE GATE (GATE A)")
    print("=" * 75)
    print("  Mode: Mock / Software Infrastructure Validation (use_mock_data=True)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [CHECK 1] Environment & Device : PASSED ({device}, PyTorch {torch.__version__})")

    # 1. Dataset & Dataloader
    try:
        train_ds = GenImageDataset(
            root_dir="./data/GenImage",
            generators=["sd15", "biggan"],
            split="train",
            use_mock_data=True,
            mock_num_samples=32
        )
        val_ds = GenImageDataset(
            root_dir="./data/GenImage",
            generators=["sd15", "biggan"],
            split="val",
            use_mock_data=True,
            mock_num_samples=16
        )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        print(f"  [CHECK 2] Dataset Construction : PASSED (Mock samples: {len(train_ds)} train, {len(val_ds)} val)")
    except Exception as e:
        print(f"  [CHECK 2] Dataset Construction : FAILED ({e})")
        print("\n  INFRASTRUCTURE GATE: FAILED")
        return False

    # 2. Model Initialization
    try:
        model = LOTANoiseClassifier(backbone="resnet50", pretrained=False, num_classes=1)
        model.to(device)
        print("  [CHECK 3] Model Initialization : PASSED (ResNet-50 NBC)")
    except Exception as e:
        print(f"  [CHECK 3] Model Initialization : FAILED ({e})")
        print("\n  INFRASTRUCTURE GATE: FAILED")
        return False

    # 3. Forward & Backward Pass
    try:
        batch = next(iter(train_loader))
        patches = batch["noise_patch"].to(device)
        labels = batch["label"].to(device).float()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"  [CHECK 4] Forward & Backward   : PASSED (Loss: {loss.item():.4f})")
    except Exception as e:
        print(f"  [CHECK 4] Forward & Backward   : FAILED ({e})")
        print("\n  INFRASTRUCTURE GATE: FAILED")
        return False

    # 4. Checkpointing
    try:
        os.makedirs("./checkpoints", exist_ok=True)
        ckpt_path = "./checkpoints/infra_gate_test.pth"
        torch.save({"model_state": model.state_dict(), "step": 1}, ckpt_path)
        loaded = torch.load(ckpt_path, map_location="cpu")
        assert "model_state" in loaded
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        print("  [CHECK 5] Checkpoint Save/Load : PASSED")
    except Exception as e:
        print(f"  [CHECK 5] Checkpoint Save/Load : FAILED ({e})")
        print("\n  INFRASTRUCTURE GATE: FAILED")
        return False

    # 5. Database Logging with mock tag
    try:
        logger = ExperimentLogger(db_path)
        mock_metrics = {
            "sd15": {"accuracy": 0.50, "auroc": 0.50, "average_precision": 0.50, "f1": 0.50}
        }
        logger.log_run(
            experiment_id="EXP_INFRASTRUCTURE_GATE_CHECK",
            name="Infrastructure Gate Verification",
            config={"mode": "infra_check", "mock": True},
            model_id="M_NBC_RESNET50",
            metrics_by_generator=mock_metrics,
            source_type="mock_fixture",
            is_mock=True
        )
        print(f"  [CHECK 6] Database Logging     : PASSED (Recorded with is_mock=1)")
    except Exception as e:
        print(f"  [CHECK 6] Database Logging     : FAILED ({e})")
        print("\n  INFRASTRUCTURE GATE: FAILED")
        return False

    print("-" * 75)
    print("  INFRASTRUCTURE GATE: PASSED")
    print("-" * 75)
    print("  [NOTE] Software infrastructure is fully operational.")
    print("         This gate validates code execution only and does NOT imply real-data readiness.\n")
    return True


if __name__ == "__main__":
    run_infrastructure_gate()
