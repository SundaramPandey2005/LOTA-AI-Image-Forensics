import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from src.experiments.database import ExperimentDatabase
from src.intelligence.templates import INTENT_REGISTRY
from src.intelligence.router import IntentRouter
from src.intelligence.grounding import GroundedExplainer


class ExperimentIntelligenceEngine:
    """
    Constrained SQL Intent Router and Natural Language Query Engine.
    Guarantees deterministic, mathematical precision by executing validated parameter queries.
    """
    def __init__(self, db_path: str = "./experiments/results/lota_experiments.db"):
        self.db = ExperimentDatabase(db_path)
        self.router = IntentRouter()
        self.explainer = GroundedExplainer()

    def route_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        return self.router.route(query)

    def execute_query(self, intent: str, params: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        if intent not in INTENT_REGISTRY:
            return pd.DataFrame(), f"Intent '{intent}' not recognized in supported template library."

        template = INTENT_REGISTRY[intent]
        metric = params.get("metric", "accuracy")
        sql_raw = template["sql"].format(metric=metric)

        sql_params = []
        if intent == "BEST_MODEL_FOR_GENERATOR":
            sql_params = [params.get("generator", "sd15")]
        elif intent == "COMPARE_MODELS":
            sql_params = [params.get("model_a", "nbc"), params.get("model_b", "ngc")]
        elif intent == "MOST_ROBUST_MODEL":
            sql_params = [params.get("perturbation_type", "jpeg"), params.get("strength", 90.0)]
        elif intent == "LARGEST_PERFORMANCE_DROP":
            sql_params = [params.get("perturbation_type", "jpeg")]

        df = self.db.query_df(sql_raw, tuple(sql_params))
        summary = self.explainer.explain(intent, params, df)

        return df, summary


def get_all_experiment_runs(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute("""
    SELECT e.experiment_id, e.name, e.timestamp, m.architecture, m.backbone
    FROM experiments e
    LEFT JOIN models m ON e.model_id = m.model_id
    ORDER BY e.timestamp DESC;
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_best_model_for_generator(conn: sqlite3.Connection, generator_name: str, metric: str = "accuracy") -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    query = f"""
    SELECT e.experiment_id, e.name, m.architecture, m.backbone, mt.{metric} as score
    FROM metrics mt
    JOIN experiments e ON mt.experiment_id = e.experiment_id
    JOIN models m ON e.model_id = m.model_id
    WHERE LOWER(mt.generator_id) = LOWER(?)
    ORDER BY score DESC
    LIMIT 5;
    """
    cursor.execute(query, (generator_name,))
    return [dict(row) for row in cursor.fetchall()]


def get_hardest_generator(conn: sqlite3.Connection, metric: str = "accuracy") -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    query = f"""
    SELECT generator_id, AVG({metric}) as avg_score, MIN({metric}) as min_score, COUNT(*) as evaluations_count
    FROM metrics
    GROUP BY generator_id
    ORDER BY avg_score ASC;
    """
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def compare_models(conn: sqlite3.Connection, exp_id_a: str, exp_id_b: str) -> Dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT generator_id, accuracy, auroc, f1 FROM metrics WHERE experiment_id = ?", (exp_id_a,))
    rows_a = {r["generator_id"]: dict(r) for r in cursor.fetchall()}

    cursor.execute("SELECT generator_id, accuracy, auroc, f1 FROM metrics WHERE experiment_id = ?", (exp_id_b,))
    rows_b = {r["generator_id"]: dict(r) for r in cursor.fetchall()}

    generators = sorted(list(set(rows_a.keys()).union(set(rows_b.keys()))))
    comparison = []
    for g in generators:
        res_a = rows_a.get(g, {})
        res_b = rows_b.get(g, {})
        comparison.append({
            "generator": g,
            "exp_a_acc": res_a.get("accuracy", None),
            "exp_b_acc": res_b.get("accuracy", None),
            "diff_acc": (res_b.get("accuracy", 0.0) - res_a.get("accuracy", 0.0)) if (res_a and res_b) else None
        })

    return {"exp_id_a": exp_id_a, "exp_id_b": exp_id_b, "comparison": comparison}


def get_most_robust_model(conn: sqlite3.Connection, transformation: str = "jpeg") -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute("""
    SELECT r.experiment_id, e.name, r.perturbation_type, r.strength, r.accuracy, r.auroc
    FROM robustness_results r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    WHERE LOWER(r.perturbation_type) LIKE LOWER(?)
    ORDER BY r.strength ASC, r.accuracy DESC;
    """, (f"%{transformation}%",))
    return [dict(row) for row in cursor.fetchall()]


def get_mgps_effect_summary(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute("""
    SELECT m.patch_size, AVG(mt.accuracy) as avg_accuracy, AVG(mt.auroc) as avg_auroc
    FROM models m
    JOIN metrics mt ON m.model_id = mt.experiment_id
    GROUP BY m.patch_size;
    """)
    return [dict(row) for row in cursor.fetchall()]
