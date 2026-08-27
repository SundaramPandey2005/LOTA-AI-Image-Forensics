import re
from typing import Tuple, Dict, Any, Optional
from .templates import INTENT_REGISTRY


class IntentRouter:
    """
    Parses user research queries and maps them to deterministic intent templates and parameters.
    """
    def __init__(self):
        self.registry = INTENT_REGISTRY

    def route(self, query: str) -> Tuple[str, Dict[str, Any]]:
        q = query.lower()

        # Metric extraction
        metric = "accuracy"
        if "auroc" in q or "auc" in q:
            metric = "auroc"
        elif "ap" in q or "average precision" in q:
            metric = "average_precision"
        elif "f1" in q:
            metric = "f1"

        # 1. Hardest generator
        if "hardest" in q or "difficult" in q or "lowest" in q:
            return "HARDEST_GENERATOR", {"metric": metric}

        # 2. Generalization gap / Unseen generator
        if "unseen" in q or "gap" in q or "generalization" in q:
            return "UNSEEN_GENERATOR_GAP", {"metric": metric}

        # 3. Robustness degradation / Drop
        if "drop" in q or "degradation" in q or "decrease" in q:
            ptype = "jpeg" if "jpeg" in q else ("blur" if "blur" in q else "noise")
            return "LARGEST_PERFORMANCE_DROP", {"perturbation_type": ptype, "metric": metric}

        if "robust" in q or "perturbation" in q:
            ptype = "jpeg" if "jpeg" in q else ("blur" if "blur" in q else "noise")
            strength = 90.0 if ptype == "jpeg" else (2.0 if ptype == "blur" else 10.0)
            return "MOST_ROBUST_MODEL", {"perturbation_type": ptype, "strength": strength, "metric": metric}

        # 4. Model comparison (e.g. NBC vs NGC)
        if "compare" in q or ("nbc" in q and "ngc" in q):
            return "COMPARE_MODELS", {"model_a": "nbc", "model_b": "ngc", "metric": metric}

        # 5. Generator specific performance
        for gen in ["biggan", "sd14", "sd15", "midjourney", "adm", "glide", "wukong", "vqdm"]:
            if gen in q:
                return "BEST_MODEL_FOR_GENERATOR", {"generator": gen, "metric": metric}

        # 6. Default leaderboard
        return "AVERAGE_BENCHMARK_LEADERBOARD", {"metric": metric}
