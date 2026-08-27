import sqlite3
from typing import Dict, Any, Tuple
from src.intelligence.templates import match_template
from src.experiments.database import ExperimentDatabase
from src.experiments import queries as qops


def route_and_execute_query(query_str: str, db_path: str = "./experiments/results/lota_experiments.db") -> Dict[str, Any]:
    """
    Route a user question to a parameterized query template,
    execute against SQLite, and return structured evidence.
    """
    template_id, params = match_template(query_str)
    if template_id is None:
        return {
            "status": "UNSUPPORTED",
            "query": query_str,
            "message": "This question does not match our pre-defined experimental templates. Supported queries include model comparisons, hardest generator identification, robustness benchmarks, and ablation summaries."
        }

    db = ExperimentDatabase(db_path=db_path)
    with db.get_connection() as conn:
        if template_id == "BEST_MODEL_FOR_GENERATOR":
            gen = params.get("generator", "sd_v14")
            records = qops.get_best_model_for_generator(conn, generator_name=gen)
            return {"status": "SUCCESS", "template_id": template_id, "params": params, "data": records}

        elif template_id == "HARDEST_GENERATOR":
            records = qops.get_hardest_generator(conn)
            return {"status": "SUCCESS", "template_id": template_id, "params": params, "data": records}

        elif template_id == "COMPARE_MODELS":
            exp_a = params.get("exp_a", "")
            exp_b = params.get("exp_b", "")
            records = qops.compare_models(conn, exp_id_a=exp_a, exp_id_b=exp_b)
            return {"status": "SUCCESS", "template_id": template_id, "params": params, "data": records}

        elif template_id == "MOST_ROBUST_MODEL":
            transform = params.get("transform", "jpeg")
            records = qops.get_most_robust_model(conn, transformation=transform)
            return {"status": "SUCCESS", "template_id": template_id, "params": params, "data": records}

        elif template_id == "MGPS_EFFECT":
            records = qops.get_mgps_effect_summary(conn)
            return {"status": "SUCCESS", "template_id": template_id, "params": params, "data": records}

        elif template_id == "LIST_ALL_EXPERIMENTS":
            records = qops.get_all_experiment_runs(conn)
            return {"status": "SUCCESS", "template_id": template_id, "params": params, "data": records}

        else:
            return {
                "status": "SUCCESS",
                "template_id": template_id,
                "params": params,
                "data": {"note": "Ablation and generalization findings are cataloged in paper_notes.md."}
            }
