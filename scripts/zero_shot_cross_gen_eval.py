"""
LOTA Zero-Shot Cross-Generator Evaluation Script
Performs cross-generator zero-shot evaluation between:
  1. E1 (Trained on BigGAN) -> Evaluated on VQDM validation data
  2. E3 (Trained on VQDM)   -> Evaluated on BigGAN validation data
Provides persistent recording to SQLite database and human-readable summary files.
"""
import os
import sys
import re
import argparse
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_parser import load_config
from src.models import create_model
from src.data.dataset import GenImageDataset
from src.training.metrics import compute_classification_metrics
from src.experiments.database import ExperimentDatabase


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
    max_samples_per_class: Optional[int] = None,
    split: str = "val_zero_shot",
    device: Optional[torch.device] = None,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Loads a model from checkpoint and evaluates on target generator validation data.
    Supports both zero-shot cross-generator and multi-generator evaluations.
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
    val_sample_limit = max_samples_per_class if max_samples_per_class is not None else data_cfg.get("max_val_samples", None)

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
        max_samples_per_class=val_sample_limit,
        use_mock_data=use_mock,
        mock_num_samples=(val_sample_limit * 2) if (use_mock and val_sample_limit) else (32 if use_mock else 64)
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

    trained_gens = cfg.get("data", {}).get("generators", cfg.get("generators", [cfg.get("generator", "unknown")]))
    if isinstance(trained_gens, str):
        trained_gens = [trained_gens]
    is_unseen = (target_generator.lower() not in [g.lower() for g in trained_gens])
    trained_gen_str = "+".join(trained_gens) if len(trained_gens) > 1 else trained_gens[0]

    result_payload = {
        "source_experiment": source_name,
        "source_experiment_id": cfg.get("experiment_name", "unknown_exp"),
        "trained_generator": trained_gen_str,
        "target_generator": target_generator,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "architecture": arch_name,
        "total_samples": len(eval_ds),
        "real_samples": real_count,
        "fake_samples": fake_count,
        "split": split,
        "source_type": "mock_fixture" if use_mock else "experimental",
        "is_mock": use_mock,
        "is_unseen": is_unseen,
        "metrics": metrics
    }

    return result_payload


def print_result_block(title: str, res: Dict[str, Any]):
    """
    Nicely formats and prints evaluation result block.
    """
    m = res["metrics"]
    eval_type = "ZERO-SHOT CROSS-GENERATOR EVALUATION" if res.get("is_unseen", True) else "MULTI-GENERATOR EVALUATION"
    print("=" * 80)
    print(f"  {eval_type}: {title}")
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



def parse_zero_shot_summary_file(summary_path: str = "./experiments/zero_shot_results_summary.txt") -> Dict[str, Any]:
    """
    Parses and validates the preserved zero-shot cross-generator evaluation summary text file.
    Raises ValueError or FileNotFoundError if the file format or required metrics are missing or inconsistent.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Zero-shot summary file not found at: '{os.path.abspath(summary_path)}'")

    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    def extract_float(pattern: str, text: str, field_name: str) -> float:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Required numeric metric '{field_name}' not found in summary file: {summary_path}")
        return float(match.group(1))

    def extract_str(pattern: str, text: str, field_name: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Required field '{field_name}' not found in summary file: {summary_path}")
        return match.group(1).strip()

    # 1. Parse E1 section
    e1_match = re.search(r"E1 BIGGAN-TRAINED -> VQDM VALIDATION RESULTS\s*(.*?)(?:E3 VQDM-TRAINED|EVALUATION PROVENANCE|$)", content, re.DOTALL | re.IGNORECASE)
    if not e1_match:
        raise ValueError(f"Could not locate 'E1 BIGGAN-TRAINED -> VQDM VALIDATION RESULTS' section in: {summary_path}")
    e1_text = e1_match.group(1)

    e1_exp_id = extract_str(r"(?:^|\n)\s*Source Experiment ID:\s*([a-zA-Z0-9_-]+)", e1_text, "E1 Experiment ID")
    e1_trained_gen = extract_str(r"(?:^|\n)\s*Trained Generator:\s*([a-zA-Z0-9_-]+)", e1_text, "E1 Trained Generator").lower()
    e1_target_gen = extract_str(r"(?:^|\n)\s*Target Evaluation Generator:\s*([a-zA-Z0-9_-]+)", e1_text, "E1 Target Generator").lower()
    e1_acc = extract_float(r"(?:^|\n)\s*Validation Accuracy:\s*([0-9.]+)", e1_text, "E1 Accuracy")
    e1_auroc = extract_float(r"(?:^|\n)\s*Validation AUROC:\s*([0-9.]+)", e1_text, "E1 AUROC")
    e1_ap = extract_float(r"(?:^|\n)\s*Average Precision:\s*([0-9.]+)", e1_text, "E1 Average Precision")
    e1_f1 = extract_float(r"(?:^|\n)\s*F1 Score:\s*([0-9.]+)", e1_text, "E1 F1 Score")
    e1_prec = extract_float(r"(?:^|\n)\s*Precision:\s*([0-9.]+)", e1_text, "E1 Precision")
    e1_rec = extract_float(r"(?:^|\n)\s*Recall:\s*([0-9.]+)", e1_text, "E1 Recall")

    # 2. Parse E3 section
    e3_match = re.search(r"E3 VQDM-TRAINED -> BIGGAN VALIDATION RESULTS\s*(.*?)(?:EVALUATION PROVENANCE|$)", content, re.DOTALL | re.IGNORECASE)
    if not e3_match:
        raise ValueError(f"Could not locate 'E3 VQDM-TRAINED -> BIGGAN VALIDATION RESULTS' section in: {summary_path}")
    e3_text = e3_match.group(1)

    e3_exp_id = extract_str(r"(?:^|\n)\s*Source Experiment ID:\s*([a-zA-Z0-9_-]+)", e3_text, "E3 Experiment ID")
    e3_trained_gen = extract_str(r"(?:^|\n)\s*Trained Generator:\s*([a-zA-Z0-9_-]+)", e3_text, "E3 Trained Generator").lower()
    e3_target_gen = extract_str(r"(?:^|\n)\s*Target Evaluation Generator:\s*([a-zA-Z0-9_-]+)", e3_text, "E3 Target Generator").lower()
    e3_acc = extract_float(r"(?:^|\n)\s*Validation Accuracy:\s*([0-9.]+)", e3_text, "E3 Accuracy")
    e3_auroc = extract_float(r"(?:^|\n)\s*Validation AUROC:\s*([0-9.]+)", e3_text, "E3 AUROC")
    e3_ap = extract_float(r"(?:^|\n)\s*Average Precision:\s*([0-9.]+)", e3_text, "E3 Average Precision")
    e3_f1 = extract_float(r"(?:^|\n)\s*F1 Score:\s*([0-9.]+)", e3_text, "E3 F1 Score")
    e3_prec = extract_float(r"(?:^|\n)\s*Precision:\s*([0-9.]+)", e3_text, "E3 Precision")
    e3_rec = extract_float(r"(?:^|\n)\s*Recall:\s*([0-9.]+)", e3_text, "E3 Recall")

    # 3. Parse Provenance section
    prov_match = re.search(r"EVALUATION PROVENANCE & STATUS\s*(.*?)$", content, re.DOTALL | re.IGNORECASE)
    if not prov_match:
        raise ValueError(f"Could not locate 'EVALUATION PROVENANCE & STATUS' section in: {summary_path}")
    prov_text = prov_match.group(1)

    status = extract_str(r"(?:^|\n)\s*Status:\s*([a-zA-Z]+)", prov_text, "Status").upper()
    source_type = extract_str(r"(?:^|\n)\s*Source Type:\s*([a-zA-Z_-]+)", prov_text, "Source Type").lower()
    mock_data_str = extract_str(r"(?:^|\n)\s*Mock Data:\s*([a-zA-Z]+)", prov_text, "Mock Data").lower()
    is_mock = mock_data_str == "true"

    if status != "COMPLETED":
        raise ValueError(f"Expected status COMPLETED, got '{status}'")
    if source_type != "experimental":
        raise ValueError(f"Expected source_type 'experimental', got '{source_type}'")
    if is_mock:
        raise ValueError("Expected is_mock to be False for real experimental results")

    return {
        "e1_to_vqdm": {
            "source_experiment": "E1 BigGAN Baseline",
            "source_experiment_id": e1_exp_id,
            "trained_generator": e1_trained_gen,
            "target_generator": e1_target_gen,
            "architecture": "nbc",
            "total_samples": 200,
            "real_samples": 100,
            "fake_samples": 100,
            "split": "val_zero_shot",
            "source_type": source_type,
            "is_mock": is_mock,
            "is_unseen": True,
            "metrics": {
                "accuracy": e1_acc,
                "auroc": e1_auroc,
                "average_precision": e1_ap,
                "f1": e1_f1,
                "precision": e1_prec,
                "recall": e1_rec
            }
        },
        "e3_to_biggan": {
            "source_experiment": "E3 VQDM Baseline",
            "source_experiment_id": e3_exp_id,
            "trained_generator": e3_trained_gen,
            "target_generator": e3_target_gen,
            "architecture": "nbc",
            "total_samples": 200,
            "real_samples": 100,
            "fake_samples": 100,
            "split": "val_zero_shot",
            "source_type": source_type,
            "is_mock": is_mock,
            "is_unseen": True,
            "metrics": {
                "accuracy": e3_acc,
                "auroc": e3_auroc,
                "average_precision": e3_ap,
                "f1": e3_f1,
                "precision": e3_prec,
                "recall": e3_rec
            }
        },
        "status": status,
        "source_type": source_type,
        "is_mock": is_mock
    }


def record_zero_shot_evaluation_results(
    db_path: str = "./experiments/results/lota_experiments.db",
    summary_path: Optional[str] = "./experiments/zero_shot_results_summary.txt",
    results_dict: Optional[Dict[str, Any]] = None,
    split: str = "val_zero_shot"
) -> bool:
    """
    Persistently records completed zero-shot cross-generator evaluation results into the SQLite database.
    Guarantees strict idempotency and provenance integrity (source_type='experimental', is_unseen=True).
    """
    if results_dict is not None:
        payload = results_dict
    elif summary_path is not None:
        payload = parse_zero_shot_summary_file(summary_path)
    else:
        raise ValueError("Either summary_path or results_dict must be provided to record results.")

    db = ExperimentDatabase(db_path)

    # 1. Ensure Model Record
    db.insert_model(
        model_id="M_NBC_RESNET50",
        architecture="nbc",
        backbone="resnet50",
        patch_size=32,
        normalization="thresholding",
        bit_planes="[0, 1, 2]"
    )

    # Process E1 Evaluation
    res_e1 = payload.get("e1_to_vqdm", {})
    e1_exp_id = res_e1.get("source_experiment_id", "biggan_constrained_baseline_e1")
    e1_target_gen = res_e1.get("target_generator", "vqdm")
    e1_metrics = res_e1.get("metrics", {})
    e1_source_type = res_e1.get("source_type", "experimental")
    e1_is_mock = res_e1.get("is_mock", False)

    # Process E3 Evaluation
    res_e3 = payload.get("e3_to_biggan", {})
    e3_exp_id = res_e3.get("source_experiment_id", "vqdm_e3_baseline")
    e3_target_gen = res_e3.get("target_generator", "biggan")
    e3_metrics = res_e3.get("metrics", {})
    e3_source_type = res_e3.get("source_type", "experimental")
    e3_is_mock = res_e3.get("is_mock", False)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Ensure E1 Experiment exists in experiments table
        cursor.execute("SELECT experiment_id FROM experiments WHERE experiment_id = ?", (e1_exp_id,))
        if not cursor.fetchone():
            e1_cfg = {
                "experiment_name": e1_exp_id,
                "generator": "biggan",
                "model": {"architecture": "nbc", "backbone": "resnet50"},
                "data": {"image_size": 256, "patch_size": 32, "bit_planes": [0, 1, 2], "normalization": "thresholding"}
            }
            db.insert_experiment(
                experiment_id=e1_exp_id,
                name="E1 BigGAN Constrained Baseline",
                model_id="M_NBC_RESNET50",
                architecture="nbc",
                config=e1_cfg,
                source_type=e1_source_type,
                is_mock=e1_is_mock,
                status="COMPLETED"
            )

        # Ensure E3 Experiment exists in experiments table
        cursor.execute("SELECT experiment_id FROM experiments WHERE experiment_id = ?", (e3_exp_id,))
        if not cursor.fetchone():
            e3_cfg = {
                "experiment_name": e3_exp_id,
                "generator": "vqdm",
                "model": {"architecture": "nbc", "backbone": "resnet50"},
                "data": {"image_size": 256, "patch_size": 32, "bit_planes": [0, 1, 2], "normalization": "thresholding"}
            }
            db.insert_experiment(
                experiment_id=e3_exp_id,
                name="E3 VQDM Baseline",
                model_id="M_NBC_RESNET50",
                architecture="nbc",
                config=e3_cfg,
                source_type=e3_source_type,
                is_mock=e3_is_mock,
                status="COMPLETED"
            )

        # Delete existing zero-shot metrics for idempotency
        cursor.execute(
            "DELETE FROM metrics WHERE experiment_id = ? AND generator_id = ? AND split = ?",
            (e1_exp_id, e1_target_gen, split)
        )
        cursor.execute(
            "DELETE FROM metrics WHERE experiment_id = ? AND generator_id = ? AND split = ?",
            (e3_exp_id, e3_target_gen, split)
        )
        conn.commit()

    # Insert zero-shot metrics with is_unseen=True
    db.insert_metrics(
        experiment_id=e1_exp_id,
        generator_id=e1_target_gen,
        metrics=e1_metrics,
        split=split,
        source_type=e1_source_type,
        is_mock=e1_is_mock,
        is_unseen=True
    )

    db.insert_metrics(
        experiment_id=e3_exp_id,
        generator_id=e3_target_gen,
        metrics=e3_metrics,
        split=split,
        source_type=e3_source_type,
        is_mock=e3_is_mock,
        is_unseen=True
    )

    print(f"[PERSISTENCE SUCCESS] Successfully recorded zero-shot cross-generator evaluation results to SQLite database ({db_path}).")
    return True


def record_e4_evaluation_results(
    db_path: str = "./experiments/results/lota_experiments.db",
    results_dict: Optional[Dict[str, Any]] = None,
    split: str = "val_multigen"
) -> bool:
    """
    Persistently records E4 multi-generator evaluation results into the SQLite database.
    Ensures model and experiment records exist and maintains strict idempotency and provenance.
    """
    if results_dict is None:
        raise ValueError("results_dict must be provided to record E4 evaluation results.")

    db = ExperimentDatabase(db_path)

    # 1. Ensure Model Record
    db.insert_model(
        model_id="M_NBC_RESNET50",
        architecture="nbc",
        backbone="resnet50",
        patch_size=32,
        normalization="thresholding",
        bit_planes="[0, 1, 2]"
    )

    with db.get_connection() as conn:
        cursor = conn.cursor()

        for key, res in results_dict.items():
            if not isinstance(res, dict) or "metrics" not in res:
                continue

            exp_id = res.get("source_experiment_id", "multi_generator_biggan_vqdm_e4")
            target_gen = res.get("target_generator", "unknown")
            metrics = res.get("metrics", {})
            source_type = res.get("source_type", "experimental")
            is_mock = res.get("is_mock", False)
            is_unseen = res.get("is_unseen", False)

            # Ensure Experiment Record exists
            cursor.execute("SELECT experiment_id FROM experiments WHERE experiment_id = ?", (exp_id,))
            if not cursor.fetchone():
                e4_cfg = {
                    "experiment_name": exp_id,
                    "generators": ["biggan", "vqdm"],
                    "model": {"architecture": "nbc", "backbone": "resnet50"},
                    "data": {"image_size": 256, "patch_size": 32, "bit_planes": [0, 1, 2], "normalization": "thresholding"}
                }
                db.insert_experiment(
                    experiment_id=exp_id,
                    name="E4 Multi-Generator (BigGAN + VQDM) Baseline",
                    model_id="M_NBC_RESNET50",
                    architecture="nbc",
                    config=e4_cfg,
                    source_type=source_type,
                    is_mock=is_mock,
                    status="COMPLETED"
                )

            # Delete existing metrics for idempotency
            cursor.execute(
                "DELETE FROM metrics WHERE experiment_id = ? AND generator_id = ? AND split = ?",
                (exp_id, target_gen, split)
            )
            conn.commit()

            # Insert evaluation metrics
            db.insert_metrics(
                experiment_id=exp_id,
                generator_id=target_gen,
                metrics=metrics,
                split=split,
                source_type=source_type,
                is_mock=is_mock,
                is_unseen=is_unseen
            )

    print(f"[PERSISTENCE SUCCESS] Successfully recorded E4 multi-generator evaluation results to SQLite database ({db_path}).")
    return True


def run_cross_generator_evaluations(
    e1_config: str = "./configs/biggan_constrained_baseline_e1.yaml",
    e1_checkpoint: str = "./checkpoints/biggan_constrained_baseline_e1_best.pth",
    e3_config: str = "./configs/vqdm_e3_baseline.yaml",
    e3_checkpoint: str = "./checkpoints/vqdm_e3_baseline_best.pth",
    data_root: str = "./data/GenImage",
    batch_size: int = 16,
    max_val_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_mock: bool = False,
    save_to_db: bool = False,
    db_path: str = "./experiments/results/lota_experiments.db"
) -> Dict[str, Dict[str, Any]]:
    """
    Executes both cross-generator zero-shot evaluations:
      1. E1 BigGAN -> VQDM
      2. E3 VQDM -> BigGAN
    Optionally persists results directly into the SQLite database.
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
        max_samples_per_class=max_val_samples,
        split="val_zero_shot",
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
        max_samples_per_class=max_val_samples,
        split="val_zero_shot",
        device=dev,
        use_mock=use_mock
    )
    print_result_block("E3 VQDM -> BigGAN", res_e3_to_biggan)
    results["e3_to_biggan"] = res_e3_to_biggan

    if save_to_db:
        record_zero_shot_evaluation_results(
            db_path=db_path,
            results_dict=results,
            split="val_zero_shot"
        )

    return results


def run_e4_evaluations(
    e4_config: str = "./configs/multi_generator_biggan_vqdm_e4.yaml",
    e4_checkpoint: str = "./checkpoints/multi_generator_biggan_vqdm_e4_best.pth",
    target_generators: Optional[List[str]] = None,
    data_root: str = "./data/GenImage",
    batch_size: int = 16,
    max_val_samples: int = 100,
    device: Optional[torch.device] = None,
    use_mock: bool = False,
    save_to_db: bool = False,
    db_path: str = "./experiments/results/lota_experiments.db",
    split: str = "val_multigen"
) -> Dict[str, Dict[str, Any]]:
    """
    Executes E4 multi-generator evaluations on target generators (default: BigGAN and VQDM validation data).
    Each target validation set consists of 100 real + 100 fake = 200 images.
    Optionally persists results directly into the SQLite database.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = target_generators or ["biggan", "vqdm"]
    print(f"\n[EVALUATION START] Initializing E4 Multi-Generator Evaluations on device: {dev}")
    print(f"  E4 Config        : {e4_config}")
    print(f"  E4 Checkpoint    : {e4_checkpoint}")
    print(f"  Target Generators: {[t.upper() for t in targets]}")
    print(f"  Samples Per Class: {max_val_samples} ({max_val_samples * 2} per target)")

    results = {}
    for idx, target in enumerate(targets, 1):
        key = f"e4_to_{target}"
        print(f"\n[RUNNING {idx}/{len(targets)}] Evaluating E4 Multi-Generator model on {target.upper()} validation data...")
        res = evaluate_cross_generator_pair(
            source_name="E4 Multi-Generator Baseline",
            target_generator=target,
            config_path=e4_config,
            checkpoint_path=e4_checkpoint,
            data_root=data_root,
            batch_size=batch_size,
            max_samples_per_class=max_val_samples,
            split=split,
            device=dev,
            use_mock=use_mock
        )
        print_result_block(f"E4 Multi-Generator -> {target.upper()}", res)
        results[key] = res

    if save_to_db:
        record_e4_evaluation_results(
            db_path=db_path,
            results_dict=results,
            split=split
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Generator & Multi-Generator Evaluation (E1, E3, E4)")
    # E1 & E3 options
    parser.add_argument("--e1-config", type=str, default="./configs/biggan_constrained_baseline_e1.yaml", help="Path to E1 config YAML")
    parser.add_argument("--e1-checkpoint", type=str, default="./checkpoints/biggan_constrained_baseline_e1_best.pth", help="Path to E1 checkpoint .pth")
    parser.add_argument("--e3-config", type=str, default="./configs/vqdm_e3_baseline.yaml", help="Path to E3 config YAML")
    parser.add_argument("--e3-checkpoint", type=str, default="./checkpoints/vqdm_e3_baseline_best.pth", help="Path to E3 checkpoint .pth")

    # E4 options
    parser.add_argument("--eval-e4", action="store_true", help="Execute E4 multi-generator evaluation")
    parser.add_argument("--e4-config", type=str, default="./configs/multi_generator_biggan_vqdm_e4.yaml", help="Path to E4 config YAML")
    parser.add_argument("--e4-checkpoint", type=str, default="./checkpoints/multi_generator_biggan_vqdm_e4_best.pth", help="Path to E4 checkpoint .pth")
    parser.add_argument("--eval-target", type=str, default=None, choices=["biggan", "vqdm"], help="Specific target generator to evaluate against (e.g. biggan or vqdm)")
    parser.add_argument("--max-val-samples", type=int, default=100, help="Max real/fake samples per class for validation evaluation (default 100 -> 200 total)")

    # General options
    parser.add_argument("--data-root", type=str, default="./data/GenImage", help="Root directory of GenImage dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for test/CI validation")
    parser.add_argument("--save-db", action="store_true", help="Persist evaluation metrics into SQLite database")
    parser.add_argument("--db-path", type=str, default="./experiments/results/lota_experiments.db", help="Path to SQLite database")
    parser.add_argument("--record-completed", action="store_true", help="Record completed zero-shot evaluation results from summary file into DB")
    parser.add_argument("--summary-path", type=str, default="./experiments/zero_shot_results_summary.txt", help="Path to summary text file")
    args = parser.parse_args()

    if args.record_completed:
        record_zero_shot_evaluation_results(
            db_path=args.db_path,
            summary_path=args.summary_path,
            split="val_zero_shot"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_e4:
        target_gens = [args.eval_target] if args.eval_target else ["biggan", "vqdm"]
        run_e4_evaluations(
            e4_config=args.e4_config,
            e4_checkpoint=args.e4_checkpoint,
            target_generators=target_gens,
            data_root=args.data_root,
            batch_size=args.batch_size,
            max_val_samples=args.max_val_samples,
            device=device,
            use_mock=args.mock,
            save_to_db=args.save_db,
            db_path=args.db_path
        )
    else:
        run_cross_generator_evaluations(
            e1_config=args.e1_config,
            e1_checkpoint=args.e1_checkpoint,
            e3_config=args.e3_config,
            e3_checkpoint=args.e3_checkpoint,
            data_root=args.data_root,
            batch_size=args.batch_size,
            device=device,
            use_mock=args.mock,
            save_to_db=args.save_db,
            db_path=args.db_path
        )


if __name__ == "__main__":
    main()

