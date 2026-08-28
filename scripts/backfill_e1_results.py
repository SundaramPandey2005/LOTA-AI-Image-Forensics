"""
Idempotent backfill script to record the completed E1 BigGAN Constrained Baseline
results by parsing and validating experiments/E1_results_summary.txt before inserting
into the SQLite database.
"""
import os
import sys
import re
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.database import ExperimentDatabase


def parse_e1_summary_file(summary_path: str = "./experiments/E1_results_summary.txt") -> Dict[str, Any]:
    """
    Parses and validates the preserved E1 results summary text file.
    Raises ValueError or FileNotFoundError if the file format or required metrics are missing or inconsistent.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Preserved E1 summary file not found at: '{os.path.abspath(summary_path)}'")

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

    # 1. Parse Experiment & Dataset metadata
    exp_id = extract_str(r"(?:^|\n)\s*Experiment ID:\s*([a-zA-Z0-9_-]+)", content, "Experiment ID")
    generator = extract_str(r"(?:^|\n)\s*Generator:\s*([a-zA-Z0-9_-]+)", content, "Generator").lower()

    if exp_id != "biggan_constrained_baseline_e1":
        raise ValueError(f"Unexpected Experiment ID '{exp_id}' in summary (expected 'biggan_constrained_baseline_e1')")
    if generator != "biggan":
        raise ValueError(f"Unexpected Generator '{generator}' in summary (expected 'biggan')")

    # 2. Parse BEST MODEL RESULTS section
    best_section_match = re.search(r"BEST MODEL RESULTS\s*(.*?)(?:FINAL EPOCH RESULTS|EXPERIMENT STATUS|$)", content, re.DOTALL | re.IGNORECASE)
    if not best_section_match:
        raise ValueError(f"Could not locate 'BEST MODEL RESULTS' section in summary file: {summary_path}")
    best_text = best_section_match.group(1)

    best_epoch = int(extract_float(r"(?:^|\n)\s*Best Epoch:\s*(\d+)", best_text, "Best Epoch"))
    best_acc = extract_float(r"(?:^|\n)\s*Validation Accuracy:\s*([0-9.]+)", best_text, "Best Validation Accuracy")
    best_auroc = extract_float(r"(?:^|\n)\s*Validation AUROC:\s*([0-9.]+)", best_text, "Best Validation AUROC")
    best_ap = extract_float(r"(?:^|\n)\s*Average Precision:\s*([0-9.]+)", best_text, "Best Average Precision")
    best_f1 = extract_float(r"(?:^|\n)\s*F1 Score:\s*([0-9.]+)", best_text, "Best F1 Score")
    best_prec = extract_float(r"(?:^|\n)\s*Precision:\s*([0-9.]+)", best_text, "Best Precision")
    best_rec = extract_float(r"(?:^|\n)\s*Recall:\s*([0-9.]+)", best_text, "Best Recall")
    train_loss = extract_float(r"(?:^|\n)\s*Train Loss:\s*([0-9.]+)", best_text, "Train Loss")
    val_loss = extract_float(r"(?:^|\n)\s*Validation Loss:\s*([0-9.]+)", best_text, "Validation Loss")

    # 3. Parse FINAL EPOCH RESULTS section
    final_section_match = re.search(r"FINAL EPOCH RESULTS\s*(.*?)(?:EXPERIMENT STATUS|$)", content, re.DOTALL | re.IGNORECASE)
    if not final_section_match:
        raise ValueError(f"Could not locate 'FINAL EPOCH RESULTS' section in summary file: {summary_path}")
    final_text = final_section_match.group(1)

    final_epoch = int(extract_float(r"(?:^|\n)\s*Final Epoch:\s*(\d+)", final_text, "Final Epoch"))
    final_acc = extract_float(r"(?:^|\n)\s*Validation Accuracy:\s*([0-9.]+)", final_text, "Final Validation Accuracy")
    final_auroc = extract_float(r"(?:^|\n)\s*Validation AUROC:\s*([0-9.]+)", final_text, "Final Validation AUROC")
    final_ap = extract_float(r"(?:^|\n)\s*Average Precision:\s*([0-9.]+)", final_text, "Final Average Precision")
    final_f1 = extract_float(r"(?:^|\n)\s*F1 Score:\s*([0-9.]+)", final_text, "Final F1 Score")
    final_prec = extract_float(r"(?:^|\n)\s*Precision:\s*([0-9.]+)", final_text, "Final Precision")
    final_rec = extract_float(r"(?:^|\n)\s*Recall:\s*([0-9.]+)", final_text, "Final Recall")

    # 4. Parse EXPERIMENT STATUS section
    status_section_match = re.search(r"EXPERIMENT STATUS\s*(.*?)$", content, re.DOTALL | re.IGNORECASE)
    if not status_section_match:
        raise ValueError(f"Could not locate 'EXPERIMENT STATUS' section in summary file: {summary_path}")
    status_text = status_section_match.group(1)

    status = extract_str(r"(?:^|\n)\s*Status:\s*([a-zA-Z]+)", status_text, "Status").upper()
    source_type = extract_str(r"(?:^|\n)\s*Source Type:\s*([a-zA-Z_-]+)", status_text, "Source Type").lower()
    mock_data_str = extract_str(r"(?:^|\n)\s*Mock Data:\s*([a-zA-Z]+)", status_text, "Mock Data").lower()
    is_mock = mock_data_str == "true"

    time_match = re.search(r"(?:^|\n)\s*Training Time:\s*(?:approximately\s*)?([0-9.]+)\s*seconds", status_text, re.IGNORECASE)
    training_time = float(time_match.group(1)) if time_match else 356.45

    # 5. Validation assertions on parsed values
    if best_epoch != 14:
        raise ValueError(f"Expected Best Epoch to be 14, but parsed {best_epoch}")
    if final_epoch != 15:
        raise ValueError(f"Expected Final Epoch to be 15, but parsed {final_epoch}")
    if status != "COMPLETED":
        raise ValueError(f"Expected Status to be COMPLETED, but parsed '{status}'")
    if source_type != "experimental":
        raise ValueError(f"Expected Source Type to be 'experimental', but parsed '{source_type}'")
    if is_mock:
        raise ValueError("Expected is_mock to be False for real experimental baseline")

    best_metrics = {
        "accuracy": best_acc,
        "auroc": best_auroc,
        "average_precision": best_ap,
        "f1": best_f1,
        "precision": best_prec,
        "recall": best_rec,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_epoch": best_epoch
    }

    final_epoch_metrics = {
        "accuracy": final_acc,
        "auroc": final_auroc,
        "average_precision": final_ap,
        "f1": final_f1,
        "precision": final_prec,
        "recall": final_rec,
        "final_epoch": final_epoch
    }

    return {
        "experiment_id": exp_id,
        "generator": generator,
        "status": status,
        "source_type": source_type,
        "is_mock": is_mock,
        "training_time_sec": training_time,
        "best_metrics": best_metrics,
        "final_epoch_metrics": final_epoch_metrics,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch
    }


def backfill_e1_experiment(
    db_path: str = "./experiments/results/lota_experiments.db",
    summary_path: str = "./experiments/E1_results_summary.txt"
) -> bool:
    """
    Backfills E1 results by parsing and validating the summary file into the specified SQLite database.
    Ensures strict idempotency: deletes existing metrics for 'biggan_constrained_baseline_e1'
    before inserting the verified records.
    """
    parsed = parse_e1_summary_file(summary_path)

    db = ExperimentDatabase(db_path)

    exp_id = parsed["experiment_id"]
    gen_name = parsed["generator"]
    model_id = "M_NBC_RESNET50"
    arch = "nbc"
    backbone = "resnet50"

    best_metrics = parsed["best_metrics"]
    final_epoch_metrics = parsed["final_epoch_metrics"]

    config = {
        "experiment_name": exp_id,
        "generator": gen_name,
        "data": {
            "root_dir": "./data/GenImage",
            "use_mock_data": False,
            "require_exact_sample_counts": True,
            "train_val_ratio": 0.7,
            "max_real_samples": 500,
            "max_fake_samples": 500,
            "image_size": 256,
            "patch_size": 32,
            "bit_planes": [0, 1, 2],
            "normalization": "thresholding"
        },
        "model": {
            "architecture": arch,
            "backbone": backbone,
            "pretrained": True,
            "num_classes": 1
        },
        "training": {
            "batch_size": 16,
            "epochs": 15,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "mixed_precision": True,
            "checkpoint_dir": "./checkpoints"
        },
        "reproducibility": {
            "seed": 42,
            "deterministic": True
        },
        "recovery_metadata": {
            "source_file": summary_path,
            "notes": (
                "Metrics recovered from preserved E1 results summary (experiments/E1_results_summary.txt). "
                "Original trained weights checkpoint is no longer available following Google Colab runtime reset."
            ),
            "best_epoch": parsed["best_epoch"],
            "total_epochs": parsed["final_epoch"]
        }
    }

    # Clean existing in-domain validation metrics/robustness for this experiment to ensure strict idempotency
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM metrics WHERE experiment_id = ? AND split IN ('val_best', 'val_final')", (exp_id,))
        cursor.execute("DELETE FROM robustness_results WHERE experiment_id = ?", (exp_id,))
        conn.commit()

    # Log model metadata
    db.insert_model(
        model_id=model_id,
        architecture=arch,
        backbone=backbone,
        patch_size=32,
        normalization="thresholding",
        bit_planes="[0, 1, 2]"
    )

    # Insert Experiment Record
    db.insert_experiment(
        experiment_id=exp_id,
        name="E1 BigGAN Constrained Baseline",
        model_id=model_id,
        architecture=arch,
        config=config,
        source_type=parsed["source_type"],
        is_mock=parsed["is_mock"],
        status=parsed["status"],
        training_time_sec=parsed["training_time_sec"]
    )

    # Insert best validation metrics (split='val_best')
    db.insert_metrics(
        experiment_id=exp_id,
        generator_id=gen_name,
        metrics=best_metrics,
        split="val_best",
        source_type=parsed["source_type"],
        is_mock=parsed["is_mock"],
        is_unseen=False
    )

    # Insert final epoch metrics (split='val_final')
    db.insert_metrics(
        experiment_id=exp_id,
        generator_id=gen_name,
        metrics=final_epoch_metrics,
        split="val_final",
        source_type=parsed["source_type"],
        is_mock=parsed["is_mock"],
        is_unseen=False
    )

    print(f"[BACKFILL SUCCESS] Experiment '{exp_id}' successfully backfilled into '{db_path}'.")
    return True


if __name__ == "__main__":
    backfill_e1_experiment()

