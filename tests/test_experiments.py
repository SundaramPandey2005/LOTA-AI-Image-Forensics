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
