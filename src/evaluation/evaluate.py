from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.training.metrics import compute_classification_metrics
from src.data.dataset import GenImageDataset, create_dataloader
from src.data.splits import ALL_GENIMAGE_GENERATORS


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        labels = batch["label"].to(device).view(-1, 1)
        noise_patch = batch["noise_patch"].to(device)
        raw_image = batch.get("raw_image", None)
        if raw_image is not None:
            raw_image = raw_image.to(device)

        if hasattr(model, "extract_image_features") or hasattr(model, "attention"):
            logits = model(noise_patch=noise_patch, raw_image=raw_image).view(-1, 1)
        else:
            logits = model(noise_patch=noise_patch).view(-1, 1)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    if len(all_preds) > 0:
        y_pred_prob = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        metrics = compute_classification_metrics(y_true, y_pred_prob)
    else:
        metrics = {"accuracy": 0.0, "ap": 0.0, "average_precision": 0.0, "auroc": 0.5, "f1": 0.0}

    return metrics


def evaluate_model_on_generators(
    model: nn.Module,
    generator_loaders: Optional[Dict[str, DataLoader]] = None,
    generators: Optional[List[str]] = None,
    root_dir: str = "./data/GenImage",
    split: str = "val",
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    use_mock_data: bool = False,
    mock_samples_per_gen: int = 32,
    architecture: str = "nbc"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a trained model against multiple generator test datasets.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(dev)
    results: Dict[str, Dict[str, float]] = {}

    if generator_loaders is not None:
        for gen_name, loader in generator_loaders.items():
            metrics = evaluate_dataloader(model, loader, device=dev)
            results[gen_name] = metrics
    else:
        eval_gens = generators or ALL_GENIMAGE_GENERATORS
        for gen in eval_gens:
            dataset = GenImageDataset(
                root_dir=root_dir,
                generators=[gen],
                split=split,
                use_mock_data=use_mock_data,
                mock_num_samples=mock_samples_per_gen
            )
            loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
            metrics = evaluate_dataloader(model, loader, device=dev)
            results[gen] = metrics
            print(f"  --> Generator: {gen:<12} | ACC: {metrics['accuracy']*100:.2f}% | AUROC: {metrics['auroc']:.4f} | AP: {metrics['ap']*100:.2f}%")

    return results
