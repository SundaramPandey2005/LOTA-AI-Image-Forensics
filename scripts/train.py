"""
LOTA Unified Training CLI
Trains NBC or NGC models based on a specified configuration file.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import torch

from src.utils.reproducibility import set_seed
from src.utils.config_parser import load_config
from src.data.dataset import GenImageDataset, create_dataloader
from src.models import create_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.evaluation.evaluate import evaluate_model_on_generators
from src.experiments.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Train LOTA AI-Generated Image Detector")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to experiment config YAML")
    parser.add_argument("--exp_name", type=str, default="lota_experiment", help="Experiment name identifier")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for verification")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Starting experiment: {args.exp_name} | Device: {device.type.upper()}")

    # 1. Dataset
    dataset_cfg = config.get("dataset", {})
    train_dataset = GenImageDataset(
        root_dir=dataset_cfg.get("root_dir", "./data/GenImage"),
        generators=config.get("training_generators", ["sd15"]),
        split="train",
        use_mock_data=args.mock,
        mock_num_samples=256
    )
    val_dataset = GenImageDataset(
        root_dir=dataset_cfg.get("root_dir", "./data/GenImage"),
        generators=config.get("training_generators", ["sd15"]),
        split="val",
        use_mock_data=args.mock,
        mock_num_samples=64
    )

    batch_size = config.get("training", {}).get("batch_size", 64)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = create_model(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("training", {}).get("learning_rate", 0.0001),
        weight_decay=config.get("training", {}).get("weight_decay", 1e-5)
    )
    epochs = config.get("training", {}).get("epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = get_loss_function()

    # 3. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        use_amp=config.get("training", {}).get("mixed_precision", True),
        early_stopping_patience=config.get("training", {}).get("early_stopping_patience", 5),
        model_name=args.exp_name
    )

    t0 = time.time()
    trainer.fit(epochs=epochs)
    elapsed = time.time() - t0

    # 4. Multi-Generator Evaluation
    eval_results = evaluate_model_on_generators(
        model=trainer.model,
        root_dir=dataset_cfg.get("root_dir", "./data/GenImage"),
        use_mock_data=args.mock,
        device=device
    )

    # 5. Log Run to SQLite
    db_path = config.get("logging", {}).get("database_path", "./experiments/results/lota_experiments.db")
    logger = ExperimentLogger(db_path)
    logger.log_run(
        experiment_id=args.exp_name.upper(),
        name=args.exp_name,
        config=config,
        model_id=f"M_{config.get('model', {}).get('architecture', 'nbc').upper()}",
        metrics_by_generator=eval_results,
        training_time_sec=elapsed
    )

    print(f"\n[DONE] Training complete for {args.exp_name} in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
