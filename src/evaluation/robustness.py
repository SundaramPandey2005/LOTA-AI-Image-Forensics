import io
from typing import Dict, List, Any, Optional, Union
from PIL import Image, ImageFilter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import extract_lota_forensic_features
from src.training.metrics import compute_classification_metrics


def apply_jpeg_compression(image_tensor: torch.Tensor, quality: int = 95) -> torch.Tensor:
    """
    Apply lossy JPEG compression in-memory.
    """
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    decompressed = Image.open(buffer).convert("RGB")
    out_np = np.array(decompressed, dtype=np.float32)
    return torch.from_numpy(out_np).permute(2, 0, 1)


def apply_gaussian_blur(image_tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian blur perturbation.
    """
    if sigma <= 0.0:
        return image_tensor

    img_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    out_np = np.array(blurred, dtype=np.float32)
    return torch.from_numpy(out_np).permute(2, 0, 1)


@torch.no_grad()
def evaluate_jpeg_robustness(
    model: nn.Module,
    samples: Union[List[Dict[str, Any]], Dataset],
    qualities: List[int] = [100, 95, 90, 85, 80, 70],
    architecture: str = "nbc",
    device: Optional[torch.device] = None
) -> Dict[int, Dict[str, float]]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(dev)
    robustness_results = {}

    for q in qualities:
        preds = []
        labels = []
        for i in range(min(len(samples), 64)):
            item = samples[i]
            raw_tensor = item["raw_image"] if isinstance(item, dict) else item[0]
            label = float(item["label"].item() if hasattr(item.get("label", 0), "item") else item.get("label", 0))

            degraded = apply_jpeg_compression(raw_tensor, quality=q)
            _, noise_patch, _ = extract_lota_forensic_features(degraded)
            noise_patch = noise_patch.unsqueeze(0).to(dev)

            if hasattr(model, "extract_image_features") or hasattr(model, "attention"):
                raw_in = degraded.unsqueeze(0).to(dev)
                logit = model(noise_patch=noise_patch, raw_image=raw_in)
            else:
                logit = model(noise_patch=noise_patch)

            prob = torch.sigmoid(logit).item()
            preds.append(prob)
            labels.append(label)

        metrics = compute_classification_metrics(np.array(labels), np.array(preds))
        robustness_results[q] = metrics

    return robustness_results


@torch.no_grad()
def evaluate_blur_robustness(
    model: nn.Module,
    samples: Union[List[Dict[str, Any]], Dataset],
    sigmas: List[float] = [0.0, 1.0, 2.0, 3.0],
    architecture: str = "nbc",
    device: Optional[torch.device] = None
) -> Dict[float, Dict[str, float]]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(dev)
    robustness_results = {}

    for sigma in sigmas:
        preds = []
        labels = []
        for i in range(min(len(samples), 64)):
            item = samples[i]
            raw_tensor = item["raw_image"] if isinstance(item, dict) else item[0]
            label = float(item["label"].item() if hasattr(item.get("label", 0), "item") else item.get("label", 0))

            degraded = apply_gaussian_blur(raw_tensor, sigma=sigma)
            _, noise_patch, _ = extract_lota_forensic_features(degraded)
            noise_patch = noise_patch.unsqueeze(0).to(dev)

            if hasattr(model, "extract_image_features") or hasattr(model, "attention"):
                raw_in = degraded.unsqueeze(0).to(dev)
                logit = model(noise_patch=noise_patch, raw_image=raw_in)
            else:
                logit = model(noise_patch=noise_patch)

            prob = torch.sigmoid(logit).item()
            preds.append(prob)
            labels.append(label)

        metrics = compute_classification_metrics(np.array(labels), np.array(preds))
        robustness_results[sigma] = metrics

    return robustness_results


def evaluate_model_robustness(
    model: nn.Module,
    base_dataset: Union[List[Dict[str, Any]], Dataset],
    device: Optional[torch.device] = None,
    architecture: str = "nbc"
) -> Dict[str, Dict[Any, Dict[str, float]]]:
    jpeg_res = evaluate_jpeg_robustness(model, base_dataset, device=device, architecture=architecture)
    blur_res = evaluate_blur_robustness(model, base_dataset, device=device, architecture=architecture)
    return {
        "jpeg": jpeg_res,
        "blur": blur_res
    }
