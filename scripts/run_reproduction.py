"""
LOTA SD v1.5 Reproduction Runner
Faithfully reproduces Section 4.2 single-generator baseline from the ICCV 2025 paper.
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
from src.evaluation.robustness import evaluate_model_robustness
from src.evaluation.visualization import plot_generalization_heatmap, plot_robustness_curves
from src.experiments.logger import ExperimentLogger


def run_sd15_reproduction(
    config_path: str = "./configs/reproduction_sd15.yaml",
    use_mock_data: bool = False
):
    config = load_config(config_path)
    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    exp_id = "EXP_LOTA_SD15_REPRODUCTION"
    print("\n" + "=" * 75)
    print(f"  EXECUTING LOTA SD v1.5 PAPER REPRODUCTION: {exp_id}")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()} | Model: NBC (ResNet-50) | Bit Planes: [0, 1, 2]")

    # 1. Dataset Setup (Train on Stable Diffusion v1.5)
    train_dataset = GenImageDataset(
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        generators=["sd15"],
        split="train",
        use_mock_data=use_mock_data,
        mock_num_samples=256
    )
    val_dataset = GenImageDataset(
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        generators=["sd15"],
        split="val",
        use_mock_data=use_mock_data,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("training", {}).get("epochs", 15)
    )
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
        model_name="lota_sd15_repro"
    )

    t_start = time.time()
    trainer.fit(epochs=config.get("training", {}).get("epochs", 15))
    t_elapsed = time.time() - t_start

    # 4. Evaluate Across All 8 GenImage Generators
    eval_results = evaluate_model_on_generators(
        model=trainer.model,
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        use_mock_data=use_mock_data,
        device=device
    )

    # 5. Robustness Testing
    robustness_results = evaluate_model_robustness(
        model=trainer.model,
        base_dataset=val_dataset,
        device=device
    )

    # 6. Log to SQLite Database
    db_path = config.get("logging", {}).get("database_path", "./experiments/results/lota_experiments.db")
    logger = ExperimentLogger(db_path)
    logger.log_run(
        experiment_id=exp_id,
        name="SD v1.5 Paper Baseline Reproduction",
        config=config,
        model_id="M_NBC_RESNET50",
        metrics_by_generator=eval_results,
        robustness_results=robustness_results,
        training_time_sec=t_elapsed
    )

    # 7. Visualizations
    plot_generalization_heatmap(
        {exp_id: eval_results},
        save_path="./experiments/visualizations/heatmap_sd15_reproduction.png"
    )
    plot_robustness_curves(
        robustness_results,
        save_path="./experiments/visualizations/robustness_sd15_reproduction.png"
    )

    print(f"\n[REPRODUCTION COMPLETE] Model and benchmark results saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOTA SD v1.5 Baseline Reproduction")
    parser.add_argument("--config", type=str, default="./configs/reproduction_sd15.yaml")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for verification")
    args = parser.parse_args()

    run_sd15_reproduction(args.config, use_mock_data=args.mock)
