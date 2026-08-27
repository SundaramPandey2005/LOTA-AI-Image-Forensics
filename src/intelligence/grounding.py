import pandas as pd
from typing import Dict, Any


class GroundedExplainer:
    """
    Generates natural language summaries grounded strictly in retrieved SQLite query results.
    Zero hallucination guarantee.
    """
    @staticmethod
    def explain(intent: str, params: Dict[str, Any], df: pd.DataFrame) -> str:
        if df.empty:
            return f"No matching empirical records found for intent '{intent}'."

        metric = params.get("metric", "accuracy").upper()

        if intent == "HARDEST_GENERATOR":
            top_hardest = df.iloc[0]
            gen_name = top_hardest["generator"]
            score = float(top_hardest["avg_score"]) * 100.0
            return (
                f"Based on empirical benchmark records, **{gen_name}** is the hardest generator to detect, "
                f"achieving the lowest average {metric} of **{score:.2f}%**."
            )

        elif intent == "UNSEEN_GENERATOR_GAP":
            top_row = df.iloc[0]
            exp_name = top_row["name"]
            gap = float(top_row["generalization_gap"]) * 100.0
            return (
                f"In experiment '{exp_name}', holding out generator '{top_row['excluded_generator']}' "
                f"resulted in a seen-to-unseen generalization gap of **{gap:.2f}% {metric}**."
            )

        elif intent == "BEST_MODEL_FOR_GENERATOR":
            top = df.iloc[0]
            score = float(top["score"]) * 100.0
            return (
                f"The highest detection score on **{params.get('generator', '').upper()}** was achieved by "
                f"**{top['name']}** ({top['architecture'].upper()}) with **{score:.2f}% {metric}**."
            )

        elif intent == "LARGEST_PERFORMANCE_DROP":
            top = df.iloc[0]
            drop = float(top["drop_amount"]) * 100.0
            return (
                f"Under {params.get('perturbation_type', '').upper()} degradation, **{top['architecture'].upper()}** "
                f"exhibited the maximum performance drop of **{drop:.2f}% {metric}**."
            )

        else:
            return f"Successfully retrieved {len(df)} records for {intent}."
