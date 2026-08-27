import os
import sqlite3
import json
from typing import Dict, Any, List, Optional
import pandas as pd


class ExperimentDatabase:
    """
    Structured SQLite Experiment Database for LOTA research with strict data provenance.
    Separates published paper reference baselines, empirical local experiments, and mock fixtures.
    """
    def __init__(self, db_path: str = "./experiments/results/lota_experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Experiments table (with source_type and is_mock provenance)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT DEFAULT 'experimental', -- 'experimental', 'published_reference', 'mock_fixture'
                is_mock BOOLEAN DEFAULT 0,
                git_commit TEXT DEFAULT 'main',
                model_id TEXT,
                architecture TEXT,
                excluded_generator TEXT,
                config_json TEXT,
                status TEXT DEFAULT 'COMPLETED',
                training_time_sec REAL DEFAULT 0.0
            );
            """)

            # 2. Models table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                architecture TEXT NOT NULL,
                backbone TEXT NOT NULL,
                patch_size INTEGER DEFAULT 32,
                normalization TEXT DEFAULT 'thresholding',
                bit_planes TEXT DEFAULT '[0, 1, 2]',
                num_params INTEGER
            );
            """)

            # 3. Generators table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS generators (
                generator_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                family TEXT NOT NULL,
                description TEXT
            );
            """)

            # 4. Metrics table (with provenance fields)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                generator_id TEXT NOT NULL,
                split TEXT DEFAULT 'test',
                source_type TEXT DEFAULT 'experimental',
                is_mock BOOLEAN DEFAULT 0,
                is_unseen BOOLEAN DEFAULT 0,
                accuracy REAL,
                auroc REAL,
                average_precision REAL,
                f1 REAL,
                precision REAL,
                recall REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            );
            """)

            # 5. Robustness Results table (with provenance fields)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS robustness_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                perturbation_type TEXT NOT NULL,
                strength REAL NOT NULL,
                source_type TEXT DEFAULT 'experimental',
                is_mock BOOLEAN DEFAULT 0,
                accuracy REAL,
                auroc REAL,
                average_precision REAL,
                f1 REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            );
            """)

            # 6. Reference Benchmarks table (strictly published paper numbers with full traceability)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reference_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_title TEXT DEFAULT 'LOTA (ICCV 2025)',
                paper_table TEXT NOT NULL,
                paper_method TEXT NOT NULL,
                training_generator TEXT NOT NULL,
                evaluation_generator TEXT NOT NULL,
                accuracy REAL,
                auroc REAL,
                average_precision REAL,
                notes TEXT
            );
            """)

            # Seed standard 8 generators
            standard_generators = [
                ("biggan", "BigGAN", "GAN", "High-capacity GAN model"),
                ("sd14", "Stable Diffusion v1.4", "Latent Diffusion", "CompVis LDM"),
                ("sd15", "Stable Diffusion v1.5", "Latent Diffusion", "RunwayML LDM"),
                ("midjourney", "Midjourney", "Proprietary Diffusion", "High-fidelity commercial generator"),
                ("adm", "ADM", "Diffusion", "Ablated Diffusion Model"),
                ("glide", "GLIDE", "Diffusion", "OpenAI Guided Diffusion"),
                ("wukong", "Wukong", "Diffusion", "Chinese text-to-image diffusion"),
                ("vqdm", "VQDM", "Vector Quantized Diffusion", "Vector Quantized Diffusion")
            ]
            for gen_id, name, family, desc in standard_generators:
                cursor.execute("""
                INSERT OR IGNORE INTO generators (generator_id, name, family, description)
                VALUES (?, ?, ?, ?);
                """, (gen_id, name, family, desc))

            conn.commit()

    def insert_experiment(
        self,
        experiment_id: str,
        name: str,
        model_id: str = "M_NBC_RESNET50",
        architecture: str = "nbc",
        config: Optional[Dict[str, Any]] = None,
        excluded_generator: Optional[str] = None,
        git_commit: str = "main",
        source_type: str = "experimental",
        is_mock: bool = False,
        training_time_sec: float = 0.0,
        status: str = "COMPLETED"
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO experiments (
                experiment_id, name, source_type, is_mock, git_commit, model_id, architecture, excluded_generator, config_json, status, training_time_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                experiment_id,
                name,
                source_type,
                1 if is_mock else 0,
                git_commit,
                model_id,
                architecture,
                excluded_generator,
                json.dumps(config or {}),
                status,
                training_time_sec
            ))
            conn.commit()

    def insert_model(
        self,
        model_id: str,
        architecture: str,
        backbone: str = "resnet50",
        patch_size: int = 32,
        normalization: str = "thresholding",
        bit_planes: str = "[0, 1, 2]",
        num_params: Optional[int] = None
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO models (
                model_id, architecture, backbone, patch_size, normalization, bit_planes, num_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (model_id, architecture, backbone, patch_size, normalization, str(bit_planes), num_params))
            conn.commit()

    def insert_metrics(
        self,
        experiment_id: str,
        generator_id: str,
        metrics: Dict[str, float],
        split: str = "test",
        source_type: str = "experimental",
        is_mock: bool = False,
        is_unseen: bool = False
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO metrics (
                experiment_id, generator_id, split, source_type, is_mock, is_unseen, accuracy, auroc, average_precision, f1, precision, recall
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                experiment_id,
                generator_id,
                split,
                source_type,
                1 if is_mock else 0,
                1 if is_unseen else 0,
                metrics.get("accuracy"),
                metrics.get("auroc"),
                metrics.get("average_precision", metrics.get("ap")),
                metrics.get("f1"),
                metrics.get("precision"),
                metrics.get("recall")
            ))
            conn.commit()

    def insert_robustness(
        self,
        experiment_id: str,
        perturbation_type: str,
        strength: float,
        metrics: Dict[str, float],
        source_type: str = "experimental",
        is_mock: bool = False
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO robustness_results (
                experiment_id, perturbation_type, strength, source_type, is_mock, accuracy, auroc, average_precision, f1
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                experiment_id,
                perturbation_type,
                float(strength),
                source_type,
                1 if is_mock else 0,
                metrics.get("accuracy"),
                metrics.get("auroc"),
                metrics.get("average_precision", metrics.get("ap")),
                metrics.get("f1")
            ))
            conn.commit()

    def insert_reference_benchmark(
        self,
        paper_table: str,
        paper_method: str,
        training_generator: str,
        evaluation_generator: str,
        accuracy: float,
        auroc: Optional[float] = None,
        average_precision: Optional[float] = None,
        notes: str = ""
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO reference_benchmarks (
                paper_table, paper_method, training_generator, evaluation_generator, accuracy, auroc, average_precision, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (paper_table, paper_method, training_generator, evaluation_generator, accuracy, auroc, average_precision, notes))
            conn.commit()

    def query_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)
