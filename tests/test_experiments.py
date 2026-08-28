import os
import sqlite3
import pytest
import tempfile

from src.experiments.database import ExperimentDatabase
from src.experiments.logger import ExperimentLogger
from src.experiments.queries import ExperimentIntelligenceEngine


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_lota_provenance.db")
    db = ExperimentDatabase(db_path)
    yield db, db_path


class TestExperimentDatabaseAndQueries:
    def test_database_initialization_and_provenance_schema(self, temp_db):
        db, _ = temp_db
        df_gens = db.query_df("SELECT * FROM generators")
        assert len(df_gens) == 8 # 8 standard GenImage generators

        # Verify provenance columns exist
        df_exp = db.query_df("PRAGMA table_info(experiments)")
        col_names = df_exp["name"].tolist()
        assert "source_type" in col_names
        assert "is_mock" in col_names

    def test_provenance_separation_and_mock_exclusion(self, temp_db):
        _, db_path = temp_db
        logger = ExperimentLogger(db_path)

        # 1. Log a real experimental run
        real_metrics = {
            "biggan": {"accuracy": 0.85, "auroc": 0.89, "average_precision": 0.88, "f1": 0.84},
            "sd14": {"accuracy": 0.82, "auroc": 0.85, "average_precision": 0.84, "f1": 0.81}
        }
        logger.log_run(
            experiment_id="EXP_REAL_RUN_1",
            name="Real Pilot Run",
            config={"model": {"architecture": "nbc"}},
            model_id="M_NBC_RESNET50",
            metrics_by_generator=real_metrics,
            source_type="experimental",
            is_mock=False
        )

        # 2. Log a mock test fixture
        mock_metrics = {
            "biggan": {"accuracy": 0.99, "auroc": 0.999, "average_precision": 0.99, "f1": 0.99},
            "sd14": {"accuracy": 0.99, "auroc": 0.999, "average_precision": 0.99, "f1": 0.99}
        }
        logger.log_run(
            experiment_id="EXP_MOCK_FIXTURE_1",
            name="Mock Demo Fixture",
            config={"model": {"architecture": "nbc"}},
            model_id="M_NBC_RESNET50",
            metrics_by_generator=mock_metrics,
            source_type="mock_fixture",
            is_mock=True
        )

        # 3. Verify intelligence router queries exclude mock fixtures by default
        engine = ExperimentIntelligenceEngine(db_path)
        intent, params = engine.route_intent("Which model achieves highest accuracy on BigGAN?")
        assert intent == "BEST_MODEL_FOR_GENERATOR"
        assert params["generator"] == "biggan"

        df_res, explanation = engine.execute_query(intent, params)
        # Result MUST NOT return EXP_MOCK_FIXTURE_1 (score 0.99), it MUST return EXP_REAL_RUN_1 (score 0.85)
        assert len(df_res) == 1
        assert df_res.iloc[0]["experiment_id"] == "EXP_REAL_RUN_1"
        assert df_res.iloc[0]["score"] == 0.85

    def test_reference_benchmarks_retrieval(self, temp_db):
        db, _ = temp_db
        db.insert_reference_benchmark(
            paper_table="Table 1",
            paper_method="LOTA-nbc",
            training_generator="sd15",
            evaluation_generator="biggan",
            accuracy=100.0,
            auroc=1.000,
            average_precision=1.000,
            notes="Paper Table 1, Row: LOTA-nbc"
        )
        df_ref = db.query_df("SELECT * FROM reference_benchmarks")
        assert len(df_ref) == 1
        assert df_ref.iloc[0]["paper_method"] == "LOTA-nbc"
        assert df_ref.iloc[0]["accuracy"] == 100.0

    def test_vqdm_e3_config_validity(self):
        """Verify that configs/vqdm_e3_baseline.yaml is valid and conforms to E1 baseline constraints."""
        import yaml
        cfg_path = "./configs/vqdm_e3_baseline.yaml"
        assert os.path.exists(cfg_path), f"Missing {cfg_path}"

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        assert cfg["experiment_name"] == "vqdm_e3_baseline"
        assert cfg["generator"] == "vqdm"
        assert cfg["model"]["architecture"] == "nbc"
        assert cfg["model"]["backbone"] == "resnet50"
        assert cfg["model"]["pretrained"] is True
        assert cfg["data"]["require_exact_sample_counts"] is True
        assert cfg["data"]["max_real_samples"] == 500
        assert cfg["data"]["max_fake_samples"] == 500
        assert cfg["data"]["image_size"] == 256
        assert cfg["data"]["patch_size"] == 32
        assert cfg["data"]["bit_planes"] == [0, 1, 2]
        assert cfg["data"]["normalization"] == "thresholding"
        assert cfg["training"]["epochs"] == 15
        assert cfg["training"]["batch_size"] == 16
        assert cfg["training"]["learning_rate"] == 0.0001
        assert cfg["training"]["optimizer"] == "adam"
        assert cfg["reproducibility"]["seed"] == 42
        assert cfg["reproducibility"]["deterministic"] is True

    def test_backfill_e1_idempotency_and_provenance(self, temp_db):
        """Verify that E1 backfill imports genuine experimental data and is strictly idempotent."""
        import json
        from scripts.backfill_e1_results import backfill_e1_experiment
        db, db_path = temp_db

        # First execution: imports E1
        success_1 = backfill_e1_experiment(db_path=db_path, summary_path="./experiments/E1_results_summary.txt")
        assert success_1 is True

        # Verify experiment row
        df_exp = db.query_df("SELECT * FROM experiments WHERE experiment_id = 'biggan_constrained_baseline_e1'")
        assert len(df_exp) == 1
        exp_row = df_exp.iloc[0]
        assert exp_row["source_type"] == "experimental"
        assert exp_row["is_mock"] == 0
        assert exp_row["status"] == "COMPLETED"
        assert exp_row["model_id"] == "M_NBC_RESNET50"
        assert exp_row["architecture"] == "nbc"

        # Verify config contains recovery notes
        cfg_loaded = json.loads(exp_row["config_json"])
        assert "recovery_metadata" in cfg_loaded
        assert "E1_results_summary.txt" in cfg_loaded["recovery_metadata"]["source_file"]

        # Verify metrics rows
        df_metrics = db.query_df("SELECT * FROM metrics WHERE experiment_id = 'biggan_constrained_baseline_e1'")
        assert len(df_metrics) == 2 # val_best and val_final

        val_best = df_metrics[df_metrics["split"] == "val_best"].iloc[0]
        assert val_best["source_type"] == "experimental"
        assert val_best["is_mock"] == 0
        assert abs(val_best["accuracy"] - 0.8700) < 1e-4
        assert abs(val_best["auroc"] - 0.94565) < 1e-4
        assert abs(val_best["average_precision"] - 0.922818) < 1e-4
        assert abs(val_best["f1"] - 0.879630) < 1e-4
        assert abs(val_best["precision"] - 0.818966) < 1e-4
        assert abs(val_best["recall"] - 0.950000) < 1e-4

        # Second execution: verify strict idempotency (no duplicate rows)
        success_2 = backfill_e1_experiment(db_path=db_path, summary_path="./experiments/E1_results_summary.txt")
        assert success_2 is True

        df_exp_after = db.query_df("SELECT * FROM experiments WHERE experiment_id = 'biggan_constrained_baseline_e1'")
        assert len(df_exp_after) == 1

        df_metrics_after = db.query_df("SELECT * FROM metrics WHERE experiment_id = 'biggan_constrained_baseline_e1'")
        assert len(df_metrics_after) == 2

    def test_parse_e1_summary_file_validation(self):
        """Verify that parse_e1_summary_file correctly parses valid summaries and rejects invalid summaries."""
        import tempfile
        from scripts.backfill_e1_results import parse_e1_summary_file

        # Test valid parsing
        res = parse_e1_summary_file("./experiments/E1_results_summary.txt")
        assert res["experiment_id"] == "biggan_constrained_baseline_e1"
        assert res["generator"] == "biggan"
        assert res["best_epoch"] == 14
        assert res["final_epoch"] == 15
        assert res["status"] == "COMPLETED"
        assert res["source_type"] == "experimental"
        assert res["is_mock"] is False
        assert abs(res["best_metrics"]["accuracy"] - 0.8700) < 1e-4
        assert abs(res["best_metrics"]["auroc"] - 0.94565) < 1e-4
        assert abs(res["final_epoch_metrics"]["accuracy"] - 0.8550) < 1e-4
        assert abs(res["final_epoch_metrics"]["auroc"] - 0.94110) < 1e-4

        # Test missing file error
        temp_dir = tempfile.mkdtemp()
        missing_file = os.path.join(temp_dir, "non_existent.txt")
        with pytest.raises(FileNotFoundError):
            parse_e1_summary_file(missing_file)

        # Test corrupted/incomplete summary error
        corrupt_file = os.path.join(temp_dir, "corrupt_summary.txt")
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("Experiment ID: biggan_constrained_baseline_e1\nGenerator: biggan\n")

        with pytest.raises(ValueError):
            parse_e1_summary_file(corrupt_file)

    def test_zero_shot_cross_gen_evaluation_mock(self):
        """Verify that zero_shot_cross_gen_eval runs end-to-end and returns expected structure."""
        import tempfile
        import torch
        from scripts.zero_shot_cross_gen_eval import run_cross_generator_evaluations
        from src.models import create_model
        from src.utils.config_parser import load_config

        temp_dir = tempfile.mkdtemp()
        mock_e1_ckpt = os.path.join(temp_dir, "mock_e1.pth")
        mock_e3_ckpt = os.path.join(temp_dir, "mock_e3.pth")

        cfg_e1 = load_config("./configs/biggan_constrained_baseline_e1.yaml")
        cfg_e3 = load_config("./configs/vqdm_e3_baseline.yaml")

        model_e1 = create_model(cfg_e1)
        model_e3 = create_model(cfg_e3)

        torch.save({"model_state_dict": model_e1.state_dict()}, mock_e1_ckpt)
        torch.save({"model_state_dict": model_e3.state_dict()}, mock_e3_ckpt)

        device = torch.device("cpu")
        res = run_cross_generator_evaluations(
            e1_config="./configs/biggan_constrained_baseline_e1.yaml",
            e1_checkpoint=mock_e1_ckpt,
            e3_config="./configs/vqdm_e3_baseline.yaml",
            e3_checkpoint=mock_e3_ckpt,
            device=device,
            use_mock=True
        )

        assert "e1_to_vqdm" in res
        assert "e3_to_biggan" in res

        for key in ("e1_to_vqdm", "e3_to_biggan"):
            r = res[key]
            assert "metrics" in r
            assert "accuracy" in r["metrics"]
            assert "auroc" in r["metrics"]
            assert "average_precision" in r["metrics"]
            assert "f1" in r["metrics"]
            assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
            assert 0.0 <= r["metrics"]["auroc"] <= 1.0

    def test_parse_zero_shot_summary_file(self):
        """Verify that parse_zero_shot_summary_file accurately parses valid summary and rejects invalid."""
        import tempfile
        from scripts.zero_shot_cross_gen_eval import parse_zero_shot_summary_file

        # Test valid parsing from actual summary file
        parsed = parse_zero_shot_summary_file("./experiments/zero_shot_results_summary.txt")
        assert "e1_to_vqdm" in parsed
        assert "e3_to_biggan" in parsed

        e1 = parsed["e1_to_vqdm"]
        assert e1["source_experiment_id"] == "biggan_constrained_baseline_e1"
        assert e1["trained_generator"] == "biggan"
        assert e1["target_generator"] == "vqdm"
        assert e1["total_samples"] == 200
        assert e1["real_samples"] == 100
        assert e1["fake_samples"] == 100
        assert abs(e1["metrics"]["accuracy"] - 0.5150) < 1e-4
        assert abs(e1["metrics"]["auroc"] - 0.5524) < 1e-4
        assert abs(e1["metrics"]["average_precision"] - 0.5407) < 1e-4
        assert abs(e1["metrics"]["f1"] - 0.3660) < 1e-4
        assert abs(e1["metrics"]["precision"] - 0.5283) < 1e-4
        assert abs(e1["metrics"]["recall"] - 0.2800) < 1e-4

        e3 = parsed["e3_to_biggan"]
        assert e3["source_experiment_id"] == "vqdm_e3_baseline"
        assert e3["trained_generator"] == "vqdm"
        assert e3["target_generator"] == "biggan"
        assert e3["total_samples"] == 200
        assert e3["real_samples"] == 100
        assert e3["fake_samples"] == 100
        assert abs(e3["metrics"]["accuracy"] - 0.4650) < 1e-4
        assert abs(e3["metrics"]["auroc"] - 0.4738) < 1e-4
        assert abs(e3["metrics"]["average_precision"] - 0.4758) < 1e-4
        assert abs(e3["metrics"]["f1"] - 0.3593) < 1e-4
        assert abs(e3["metrics"]["precision"] - 0.4478) < 1e-4
        assert abs(e3["metrics"]["recall"] - 0.3000) < 1e-4

        # Test error handling on missing file
        temp_dir = tempfile.mkdtemp()
        missing_file = os.path.join(temp_dir, "missing_zero_shot.txt")
        with pytest.raises(FileNotFoundError):
            parse_zero_shot_summary_file(missing_file)

        # Test error handling on corrupt file
        corrupt_file = os.path.join(temp_dir, "corrupt_zero_shot.txt")
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("Invalid Zero Shot Summary Content\n")

        with pytest.raises(ValueError):
            parse_zero_shot_summary_file(corrupt_file)

    def test_record_zero_shot_evaluation_results_idempotency_and_provenance(self, temp_db):
        """Verify that zero-shot evaluation recording writes valid provenance records and is idempotent."""
        from scripts.zero_shot_cross_gen_eval import record_zero_shot_evaluation_results
        db, db_path = temp_db

        # First recording
        success_1 = record_zero_shot_evaluation_results(
            db_path=db_path,
            summary_path="./experiments/zero_shot_results_summary.txt"
        )
        assert success_1 is True

        # Verify experiments table
        df_exp = db.query_df("SELECT * FROM experiments WHERE experiment_id IN ('biggan_constrained_baseline_e1', 'vqdm_e3_baseline')")
        assert len(df_exp) == 2
        for _, row in df_exp.iterrows():
            assert row["source_type"] == "experimental"
            assert row["is_mock"] == 0
            assert row["status"] == "COMPLETED"

        # Verify metrics table
        df_metrics = db.query_df("SELECT * FROM metrics WHERE split = 'val_zero_shot'")
        assert len(df_metrics) == 2

        # Verify E1 -> VQDM
        row_e1 = df_metrics[df_metrics["experiment_id"] == "biggan_constrained_baseline_e1"].iloc[0]
        assert row_e1["generator_id"] == "vqdm"
        assert row_e1["split"] == "val_zero_shot"
        assert row_e1["is_unseen"] == 1
        assert row_e1["source_type"] == "experimental"
        assert row_e1["is_mock"] == 0
        assert abs(row_e1["accuracy"] - 0.5150) < 1e-4
        assert abs(row_e1["auroc"] - 0.5524) < 1e-4
        assert abs(row_e1["average_precision"] - 0.5407) < 1e-4
        assert abs(row_e1["f1"] - 0.3660) < 1e-4
        assert abs(row_e1["precision"] - 0.5283) < 1e-4
        assert abs(row_e1["recall"] - 0.2800) < 1e-4

        # Verify E3 -> BigGAN
        row_e3 = df_metrics[df_metrics["experiment_id"] == "vqdm_e3_baseline"].iloc[0]
        assert row_e3["generator_id"] == "biggan"
        assert row_e3["split"] == "val_zero_shot"
        assert row_e3["is_unseen"] == 1
        assert row_e3["source_type"] == "experimental"
        assert row_e3["is_mock"] == 0
        assert abs(row_e3["accuracy"] - 0.4650) < 1e-4
        assert abs(row_e3["auroc"] - 0.4738) < 1e-4
        assert abs(row_e3["average_precision"] - 0.4758) < 1e-4
        assert abs(row_e3["f1"] - 0.3593) < 1e-4
        assert abs(row_e3["precision"] - 0.4478) < 1e-4
        assert abs(row_e3["recall"] - 0.3000) < 1e-4

        # Second recording: verify idempotency
        success_2 = record_zero_shot_evaluation_results(
            db_path=db_path,
            summary_path="./experiments/zero_shot_results_summary.txt"
        )
        assert success_2 is True

        df_metrics_after = db.query_df("SELECT * FROM metrics WHERE split = 'val_zero_shot'")
        assert len(df_metrics_after) == 2

    def test_zero_shot_cross_gen_evaluation_mock_with_db_save(self, temp_db):
        """Verify that zero-shot evaluation can execute with mock data and save directly to SQLite."""
        import tempfile
        import torch
        from scripts.zero_shot_cross_gen_eval import run_cross_generator_evaluations
        from src.models import create_model
        from src.utils.config_parser import load_config
        db, db_path = temp_db

        temp_dir = tempfile.mkdtemp()
        mock_e1_ckpt = os.path.join(temp_dir, "mock_e1.pth")
        mock_e3_ckpt = os.path.join(temp_dir, "mock_e3.pth")

        cfg_e1 = load_config("./configs/biggan_constrained_baseline_e1.yaml")
        cfg_e3 = load_config("./configs/vqdm_e3_baseline.yaml")

        model_e1 = create_model(cfg_e1)
        model_e3 = create_model(cfg_e3)

        torch.save({"model_state_dict": model_e1.state_dict()}, mock_e1_ckpt)
        torch.save({"model_state_dict": model_e3.state_dict()}, mock_e3_ckpt)

        device = torch.device("cpu")
        res = run_cross_generator_evaluations(
            e1_config="./configs/biggan_constrained_baseline_e1.yaml",
            e1_checkpoint=mock_e1_ckpt,
            e3_config="./configs/vqdm_e3_baseline.yaml",
            e3_checkpoint=mock_e3_ckpt,
            device=device,
            use_mock=True,
            save_to_db=True,
            db_path=db_path
        )

        assert "e1_to_vqdm" in res
        assert "e3_to_biggan" in res

        df_metrics = db.query_df("SELECT * FROM metrics WHERE split = 'val_zero_shot'")
        assert len(df_metrics) == 2
        for _, row in df_metrics.iterrows():
            assert row["is_mock"] == 1
            assert row["is_unseen"] == 1
            assert row["source_type"] == "mock_fixture"

    def test_e4_multi_generator_db_logging_and_provenance(self, temp_db):
        """Verify that E4 multi-generator experiment is logged with source_type='experimental' and is_mock=0."""
        _, db_path = temp_db
        logger = ExperimentLogger(db_path)

        e4_metrics = {
            "accuracy": 0.88,
            "auroc": 0.92,
            "average_precision": 0.91,
            "f1": 0.87,
            "precision": 0.89,
            "recall": 0.85
        }

        logger.log_run(
            experiment_id="multi_generator_biggan_vqdm_e4",
            name="E4 Multi-Generator (BigGAN + VQDM) Baseline",
            config={"data": {"generators": ["biggan", "vqdm"], "samples_per_generator_class": 250}},
            model_id="M_NBC_RESNET50",
            metrics_by_generator={
                "multi_generator": e4_metrics,
                "biggan": e4_metrics,
                "vqdm": e4_metrics
            },
            split="val_best",
            source_type="experimental",
            is_mock=False,
            training_time_sec=120.0
        )

        db = ExperimentDatabase(db_path)
        df_exp = db.query_df("SELECT * FROM experiments WHERE experiment_id = 'multi_generator_biggan_vqdm_e4'")
        assert len(df_exp) == 1
        assert df_exp.iloc[0]["source_type"] == "experimental"
        assert df_exp.iloc[0]["is_mock"] == 0
        assert df_exp.iloc[0]["status"] == "COMPLETED"

        df_metrics = db.query_df("SELECT * FROM metrics WHERE experiment_id = 'multi_generator_biggan_vqdm_e4'")
        assert len(df_metrics) == 3
        for _, row in df_metrics.iterrows():
            assert row["source_type"] == "experimental"
            assert row["is_mock"] == 0
            assert row["accuracy"] == 0.88
            assert row["auroc"] == 0.92





