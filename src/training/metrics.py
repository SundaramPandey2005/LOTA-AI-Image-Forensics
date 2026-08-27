from typing import Dict, List, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List[float], List[int]],
    y_pred_prob: Union[np.ndarray, torch.Tensor, List[float]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.
    Guarantees robust edge-case handling for AUROC, AP, precision, recall, and F1.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    elif isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
    elif isinstance(y_pred_prob, list):
        y_pred_prob = np.array(y_pred_prob)

    y_true = np.asarray(y_true, dtype=np.int32).flatten()
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float32).flatten()

    if len(y_true) == 0 or len(y_pred_prob) == 0:
        return {
            "accuracy": 0.0,
            "ap": 0.5,
            "average_precision": 0.5,
            "auroc": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }

    # Replace any NaN or Inf in probabilities with 0.5
    if np.isnan(y_pred_prob).any() or np.isinf(y_pred_prob).any():
        y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.5, posinf=1.0, neginf=0.0)

    y_pred = (y_pred_prob >= threshold).astype(np.int32)
    acc = float(accuracy_score(y_true, y_pred))

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            auroc_val = float(roc_auc_score(y_true, y_pred_prob))
            auroc = 0.5 if (np.isnan(auroc_val) or np.isinf(auroc_val)) else auroc_val
        except Exception:
            auroc = 0.5

        try:
            ap_val = float(average_precision_score(y_true, y_pred_prob))
            ap = 0.5 if (np.isnan(ap_val) or np.isinf(ap_val)) else ap_val
        except Exception:
            ap = 0.5
    else:
        # Edge case: single class present in evaluation set
        auroc = 0.5
        ap = 0.5

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "ap": ap,
        "average_precision": ap,
        "auroc": auroc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }


# Aliases
compute_binary_metrics = compute_classification_metrics


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_trues: List[float] = []
        self.y_probs: List[float] = []
        self.losses: List[float] = []

    def update(self, y_true: torch.Tensor, logits: torch.Tensor, loss: float):
        probs = torch.sigmoid(logits)
        self.y_trues.extend(y_true.detach().cpu().flatten().tolist())
        self.y_probs.extend(probs.detach().cpu().flatten().tolist())
        self.losses.append(loss)

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        metrics = compute_classification_metrics(
            np.array(self.y_trues),
            np.array(self.y_probs),
            threshold=threshold
        )
        metrics["loss"] = float(np.mean(self.losses)) if self.losses else 0.0
        return metrics
