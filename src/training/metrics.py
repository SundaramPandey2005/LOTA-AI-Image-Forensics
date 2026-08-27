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
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_prob: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()

    y_true = y_true.astype(np.int32).flatten()
    y_pred_prob = y_pred_prob.astype(np.float32).flatten()
    y_pred = (y_pred_prob >= threshold).astype(np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_pred_prob))
        except Exception:
            auroc = 0.5
        try:
            ap = float(average_precision_score(y_true, y_pred_prob))
        except Exception:
            ap = 0.5
    else:
        auroc = 1.0 if np.all(y_pred == y_true) else 0.5
        ap = 1.0 if np.all(y_pred == y_true) else 0.5

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
