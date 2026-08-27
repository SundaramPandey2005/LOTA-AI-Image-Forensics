"""
Test Fixture Seeder for Development & Streamlit Demo Mock Mode.
All data inserted by this script is strictly marked with:
    source_type = 'mock_fixture'
    is_mock = 1
This data is excluded from production research analysis by default.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.logger import ExperimentLogger


def seed_test_fixture(db_path: str = "./experiments/results/lota_experiments.db"):
    logger = ExperimentLogger(db_path)

    print(f"[TEST FIXTURE] Seeding mock test fixtures (is_mock=1) into {db_path}...")

    # Mock fixture metrics for UI testing
    mock_metrics = {
        "biggan": {"accuracy": 0.95, "auroc": 0.96, "average_precision": 0.95, "f1": 0.94},
        "sd14": {"accuracy": 0.92, "auroc": 0.94, "average_precision": 0.93, "f1": 0.91},
        "midjourney": {"accuracy": 0.88, "auroc": 0.90, "average_precision": 0.89, "f1": 0.87},
        "adm": {"accuracy": 0.91, "auroc": 0.93, "average_precision": 0.92, "f1": 0.90}
    }
    mock_robustness = {
        "jpeg": {
            100.0: {"accuracy": 0.95, "auroc": 0.96, "average_precision": 0.95, "f1": 0.94},
            90.0: {"accuracy": 0.90, "auroc": 0.91, "average_precision": 0.90, "f1": 0.89},
            70.0: {"accuracy": 0.80, "auroc": 0.82, "average_precision": 0.81, "f1": 0.79}
        }
    }

    logger.log_run(
        experiment_id="EXP_MOCK_DEMO_FIXTURE",
        name="[TEST FIXTURE] Mock NBC Demo Run",
        config={"model": {"architecture": "nbc"}},
        model_id="M_MOCK_NBC",
        metrics_by_generator=mock_metrics,
        robustness_results=mock_robustness,
        source_type="mock_fixture",
        is_mock=True,
        training_time_sec=12.0
    )

    print("[SUCCESS] Test fixtures seeded with source_type='mock_fixture' and is_mock=1!")


if __name__ == "__main__":
    seed_test_fixture()
