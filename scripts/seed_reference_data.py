"""
Database Pre-Seeder for Published ICCV 2025 LOTA Paper Benchmarks.
Strictly records verified published reference figures into `reference_benchmarks`.
Every entry is fully traceable to its exact Table, Method, and Condition in the ICCV 2025 paper.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.database import ExperimentDatabase


def seed_reference_benchmarks(db_path: str = "./experiments/results/lota_experiments.db"):
    db = ExperimentDatabase(db_path)

    print(f"[REFERENCE DATA] Seeding published ICCV 2025 benchmark data into {db_path}...")

    # Published Table 1 & Table 2 metrics from LOTA ICCV 2025 Paper
    # Format: (paper_table, paper_method, training_generator, evaluation_generator, accuracy, auroc, ap, notes)
    verified_reference_data = [
        # --- Paper Table 1: In-Domain & Cross-Generator Detection (Trained on SD v1.5) ---
        ("Table 1", "LOTA-nbc", "sd15", "biggan", 100.0, 1.000, 1.000, "Paper Table 1, Row: LOTA-nbc"),
        ("Table 1", "LOTA-nbc", "sd15", "sd14", 99.9, 0.999, 0.999, "Paper Table 1, Row: LOTA-nbc"),
        ("Table 1", "LOTA-nbc", "sd15", "sd15", 99.9, 0.999, 0.999, "Paper Table 1, Row: LOTA-nbc"),
        ("Table 1", "LOTA-nbc", "sd15", "midjourney", 93.1, 0.962, 0.958, "Paper Table 1, Row: LOTA-nbc"),
        ("Table 1", "LOTA-nbc", "sd15", "adm", 99.7, 0.999, 0.999, "Paper Table 1, Row: LOTA-nbc (Corrected from 98.5 -> 99.7)"),
        ("Table 1", "LOTA-nbc", "sd15", "glide", 100.0, 1.000, 1.000, "Paper Table 1, Row: LOTA-nbc"),
        ("Table 1", "LOTA-nbc", "sd15", "wukong", 99.8, 0.999, 0.999, "Paper Table 1, Row: LOTA-nbc"),
        ("Table 1", "LOTA-nbc", "sd15", "vqdm", 99.7, 0.998, 0.998, "Paper Table 1, Row: LOTA-nbc"),

        # --- Paper Table 2: LOGO Generalization Setting (LOTA-nbc Row) ---
        ("Table 2", "LOTA-nbc", "LOGO (remaining 3)", "biggan", 86.5, None, None, "Paper Table 2, Row: LOTA-nbc (Held-Out BigGAN)"),
        ("Table 2", "LOTA-nbc", "LOGO (remaining 3)", "sd14", 99.7, None, None, "Paper Table 2, Row: LOTA-nbc (Held-Out SD1.4)"),
        ("Table 2", "LOTA-nbc", "LOGO (remaining 3)", "midjourney", 88.4, None, None, "Paper Table 2, Row: LOTA-nbc (Held-Out Midjourney)"),
        ("Table 2", "LOTA-nbc", "LOGO (remaining 3)", "adm", 97.8, None, None, "Paper Table 2, Row: LOTA-nbc (Held-Out ADM)"),

        # --- Paper Table 2: LOGO Generalization Setting (LOTA-ngc Row) ---
        ("Table 2", "LOTA-ngc", "LOGO (remaining 3)", "biggan", 88.1, None, None, "Paper Table 2, Row: LOTA-ngc (Held-Out BigGAN)"),
        ("Table 2", "LOTA-ngc", "LOGO (remaining 3)", "sd14", 99.8, None, None, "Paper Table 2, Row: LOTA-ngc (Held-Out SD1.4)"),
        ("Table 2", "LOTA-ngc", "LOGO (remaining 3)", "midjourney", 90.4, None, None, "Paper Table 2, Row: LOTA-ngc (Held-Out Midjourney)"),
        ("Table 2", "LOTA-ngc", "LOGO (remaining 3)", "adm", 98.0, None, None, "Paper Table 2, Row: LOTA-ngc (Held-Out ADM)"),
    ]

    for table, method, train_gen, eval_gen, acc, auroc, ap, notes in verified_reference_data:
        db.insert_reference_benchmark(
            paper_table=table,
            paper_method=method,
            training_generator=train_gen,
            evaluation_generator=eval_gen,
            accuracy=acc,
            auroc=auroc,
            average_precision=ap,
            notes=notes
        )

    print("[SUCCESS] All 16 verified ICCV 2025 published reference benchmarks seeded into reference_benchmarks table!")


if __name__ == "__main__":
    seed_reference_benchmarks()
