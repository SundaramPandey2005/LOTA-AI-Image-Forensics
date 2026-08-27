"""
Leave-One-Generator-Out (LOGO) Experiment Runner for LOTA
Executes controlled cross-generator generalization training and evaluations.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import torch

from src.utils.reproducibility import set_seed
from src.utils.config_parser import load_config
from src.data.splits import GenImageSplits, LOGO_REPRESENTATIVE_GENERATORS
from src.data.dataset import GenImageDataset, create_dataloader
from src.models import create_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.evaluation.evaluate import evaluate_model_on_generators
from src.evaluation.robustness import evaluate_model_robustness
from src.evaluation.visualization import plot_generalization_heatmap, plot_robustness_curves
from src.experiments.logger import ExperimentLogger


def run_logo_rotation(
    excluded_generator: str,
    config_path: str = "./configs/logo_matrix.yaml",
    use_mock_data: bool = False
):
    config = load_config(config_path)
    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    splits = GenImageSplits(root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"), representative_only=True)
    train_gens, held_out = splits.get_logo_split(excluded_generator)

    exp_id = f"EXP_LOGO_EXCL_{excluded_generator.upper()}"
    print("\n" + "=" * 75)
    print(f"  EXECUTING LOGO ROTATION: {exp_id}")
    print(f"  Training Generators : {train_gens}")
    print(f"  Held-Out Generator  : {held_out} (Strict Zero-Shot Test)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset Setup
    train_dataset = GenImageDataset(
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        generators=train_gens,
        split="train",
        use_mock_data=use_mock_data,
        mock_num_samples=256
    )
    val_dataset = GenImageDataset(
        root_dir=config.get("dataset", {}).get("root_dir", "./data/GenImage"),
        generators=train_gens,
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
        T_max=config.get("training", {}).get("epochs", 10)
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
        model_name=f"lota_logo_{excluded_generator}"
    )

    t_start = time.time()
    trainer.fit(epochs=config.get("training", {}).get("epochs", 10))
    t_elapsed = time.time() - t_start

    # 4. Full 8-Generator Evaluation Matrix
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
        name=f"LOGO Exclude {excluded_generator.capitalize()}",
        config=config,
        model_id=f"M_{config.get('model', {}).get('architecture', 'nbc').upper()}_RESNET50",
        metrics_by_generator=eval_results,
        robustness_results=robustness_results,
        training_time_sec=t_elapsed,
        excluded_generator=excluded_generator
    )

    # 7. Visualizations
    plot_generalization_heatmap(
        {exp_id: eval_results},
        save_path=f"./experiments/visualizations/heatmap_{exp_id}.png"
    )
    plot_robustness_curves(
        robustness_results,
        save_path=f"./experiments/visualizations/robustness_{exp_id}.png"
    )

    print(f"\n[LOGO COMPLETE] Run '{exp_id}' successfully executed and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOTA LOGO Experiment")
    parser.add_argument("--excluded_generator", type=str, default="midjourney", choices=LOGO_REPRESENTATIVE_GENERATORS)
    parser.add_argument("--config", type=str, default="./configs/logo_matrix.yaml")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for verification")
    args = parser.parse_args()

    run_logo_rotation(args.excluded_generator, args.config, use_mock_data=args.mock)
