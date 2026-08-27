from typing import Dict, Any

INTENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BEST_MODEL_FOR_GENERATOR": {
        "description": "Finds the top performing model architectures for a specific AI generator from empirical experiments.",
        "sql": """
            SELECT m.experiment_id, e.architecture, e.name, m.generator_id, m.{metric} AS score
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE LOWER(m.generator_id) = LOWER(?)
              AND m.is_mock = 0
            ORDER BY score DESC
            LIMIT 5;
        """,
        "params": ["generator", "metric"]
    },
    "HARDEST_GENERATOR": {
        "description": "Ranks generators by average detection score to find the most challenging out-of-domain generator.",
        "sql": """
            SELECT g.name AS generator, g.family, AVG(m.{metric}) AS avg_score, MIN(m.{metric}) AS min_score
            FROM metrics m
            JOIN generators g ON m.generator_id = g.generator_id
            WHERE m.is_mock = 0
            GROUP BY m.generator_id
            ORDER BY avg_score ASC;
        """,
        "params": ["metric"]
    },
    "COMPARE_MODELS": {
        "description": "Directly compares two model architectures (e.g. NBC vs NGC) across all generator splits.",
        "sql": """
            SELECT m.generator_id,
                   MAX(CASE WHEN e.architecture = ? THEN m.{metric} END) AS model_a_score,
                   MAX(CASE WHEN e.architecture = ? THEN m.{metric} END) AS model_b_score
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE m.is_mock = 0
            GROUP BY m.generator_id;
        """,
        "params": ["model_a", "model_b", "metric"]
    },
    "MOST_ROBUST_MODEL": {
        "description": "Finds the model that maintains highest detection fidelity under specified image perturbation.",
        "sql": """
            SELECT r.experiment_id, e.architecture, r.perturbation_type, r.strength, r.{metric} AS score
            FROM robustness_results r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            WHERE LOWER(r.perturbation_type) = LOWER(?) AND r.strength = ?
              AND r.is_mock = 0
            ORDER BY score DESC;
        """,
        "params": ["perturbation_type", "strength", "metric"]
    },
    "LARGEST_PERFORMANCE_DROP": {
        "description": "Computes the maximum performance drop from unperturbed baseline to worst perturbation.",
        "sql": """
            SELECT r.experiment_id, e.architecture, r.perturbation_type,
                   MAX(CASE WHEN r.strength = 100 OR r.strength = 0 THEN r.{metric} END) AS clean_score,
                   MIN(r.{metric}) AS worst_score,
                   (MAX(CASE WHEN r.strength = 100 OR r.strength = 0 THEN r.{metric} END) - MIN(r.{metric})) AS drop_amount
            FROM robustness_results r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            WHERE LOWER(r.perturbation_type) = LOWER(?)
              AND r.is_mock = 0
            GROUP BY r.experiment_id
            ORDER BY drop_amount DESC;
        """,
        "params": ["perturbation_type", "metric"]
    },
    "AVERAGE_BENCHMARK_LEADERBOARD": {
        "description": "Overall experiment leaderboard ranked by average score across all 8 GenImage test splits.",
        "sql": """
            SELECT e.experiment_id, e.name, e.architecture, AVG(m.{metric}) AS avg_score, AVG(m.auroc) AS avg_auroc
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE m.is_mock = 0
            GROUP BY e.experiment_id
            ORDER BY avg_score DESC;
        """,
        "params": ["metric"]
    },
    "UNSEEN_GENERATOR_GAP": {
        "description": "Calculates the empirical generalization gap between in-domain seen and out-of-domain unseen generators.",
        "sql": """
            SELECT e.experiment_id, e.name, e.excluded_generator,
                   AVG(CASE WHEN m.is_unseen = 0 THEN m.{metric} END) AS seen_avg,
                   AVG(CASE WHEN m.is_unseen = 1 THEN m.{metric} END) AS unseen_avg,
                   (AVG(CASE WHEN m.is_unseen = 0 THEN m.{metric} END) - AVG(CASE WHEN m.is_unseen = 1 THEN m.{metric} END)) AS generalization_gap
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE e.excluded_generator IS NOT NULL
              AND m.is_mock = 0
            GROUP BY e.experiment_id;
        """,
        "params": ["metric"]
    },
    "PUBLISHED_PAPER_REFERENCE": {
        "description": "Retrieves published ICCV 2025 paper benchmark reference numbers for comparison.",
        "sql": """
            SELECT method_name, evaluation_setting, generator_id, accuracy, auroc, average_precision, notes
            FROM reference_benchmarks
            ORDER BY method_name, evaluation_setting, generator_id;
        """,
        "params": []
    }
}


def get_query_template(intent_name: str) -> Dict[str, Any]:
    if intent_name not in INTENT_REGISTRY:
        raise ValueError(f"Intent '{intent_name}' not found in registry.")
    return INTENT_REGISTRY[intent_name]
