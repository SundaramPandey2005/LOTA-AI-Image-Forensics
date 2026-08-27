from typing import Dict, Any, Optional, Union
from src.experiments.database import ExperimentDatabase


class ExperimentLogger:
    """
    High-level Logger that records training, model architectures, cross-generator evaluations,
    and robustness degradation curves to the SQLite database with strict provenance tags.
    """
    def __init__(self, db_path: str = "./experiments/results/lota_experiments.db"):
        self.db = ExperimentDatabase(db_path=db_path)

    def log_run(
        self,
        experiment_id: Optional[str] = None,
        name: str = "lota_experiment",
        config: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        metrics_by_generator: Optional[Dict[str, Dict[str, float]]] = None,
        robustness_results: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None,
        training_time_sec: float = 0.0,
        excluded_generator: Optional[str] = None,
        source_type: str = "experimental",
        is_mock: bool = False,
        split: str = "test",
        exp_id: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        robustness_metrics: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None,
        notes: str = ""
    ):
        # Resolve aliases
        final_exp_id = experiment_id or exp_id or "EXP_UNNAMED"
        final_config = config or {}
        final_metrics = metrics_by_generator or eval_metrics or {}
        final_robustness = robustness_results or robustness_metrics or {}

        # 1. Resolve model attributes
        if isinstance(model_info, dict):
            arch = model_info.get("architecture", final_config.get("model", {}).get("architecture", "nbc"))
            backbone = model_info.get("backbone", final_config.get("model", {}).get("backbone", "resnet50"))
            m_id = model_id or model_info.get("model_id", f"M_{arch.upper()}")
        else:
            arch = final_config.get("model", {}).get("architecture", "nbc")
            backbone = final_config.get("model", {}).get("backbone", "resnet50")
            m_id = model_id or f"M_{arch.upper()}"

        patch_size = final_config.get("forensic", {}).get("patch_size", 32)
        normalization = final_config.get("forensic", {}).get("normalization", "thresholding")
        bit_planes = str(final_config.get("forensic", {}).get("bit_planes", [0, 1, 2]))

        # 2. Insert Model
        self.db.insert_model(
            model_id=m_id,
            architecture=arch,
            backbone=backbone,
            patch_size=patch_size,
            normalization=normalization,
            bit_planes=bit_planes
        )

        # 3. Insert Experiment with strict provenance
        self.db.insert_experiment(
            experiment_id=final_exp_id,
            name=name,
            model_id=m_id,
            architecture=arch,
            config=final_config,
            excluded_generator=excluded_generator,
            source_type=source_type,
            is_mock=is_mock,
            training_time_sec=training_time_sec,
            status="COMPLETED"
        )

        # 4. Insert Metrics
        for gen_name, metrics in final_metrics.items():
            is_unseen = (gen_name.lower() == str(excluded_generator).lower()) if excluded_generator else False
            self.db.insert_metrics(
                experiment_id=final_exp_id,
                generator_id=gen_name,
                split=split,
                source_type=source_type,
                is_mock=is_mock,
                metrics=metrics,
                is_unseen=is_unseen
            )

        # 5. Insert Robustness
        if final_robustness:
            for ptype, s_dict in final_robustness.items():
                for strength, m in s_dict.items():
                    self.db.insert_robustness(
                        experiment_id=final_exp_id,
                        perturbation_type=ptype,
                        strength=float(strength),
                        source_type=source_type,
                        is_mock=is_mock,
                        metrics=m
                    )

        print(f"[EXPERIMENT LOGGED] Successfully recorded run '{final_exp_id}' [source: {source_type}, mock: {is_mock}] to SQLite database ({self.db.db_path}).")
