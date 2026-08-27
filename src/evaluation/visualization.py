import os
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_logo_heatmap(
    matrix_data: Union[np.ndarray, Dict[str, Any]],
    train_labels: Optional[List[str]] = None,
    test_labels: Optional[List[str]] = None,
    save_path: Optional[str] = "./experiments/visualizations/logo_generalization_heatmap.png",
    title: str = "Cross-Generator Generalization Matrix (Accuracy %)"
):
    """
    Generate Cross-Generator Generalization Heatmap (matching Figure 5).
    """
    if isinstance(matrix_data, dict):
        # Convert nested dict to matrix
        train_keys = list(matrix_data.keys())
        first_val = matrix_data[train_keys[0]]
        test_keys = list(first_val.keys()) if isinstance(first_val, dict) else train_keys
        mat = np.zeros((len(train_keys), len(test_keys)), dtype=np.float32)
        for i, tr in enumerate(train_keys):
            for j, te in enumerate(test_keys):
                val = matrix_data[tr].get(te, 0.0)
                if isinstance(val, dict):
                    val = val.get("accuracy", 0.0) * 100.0
                elif isinstance(val, (float, int)) and val <= 1.0:
                    val = val * 100.0
                mat[i, j] = val
        data = mat
        t_labels = train_keys
        e_labels = test_keys
    else:
        data = matrix_data
        t_labels = train_labels or ["Model"]
        e_labels = test_labels or ["Generator"]

    plt.figure(figsize=(10, 6), dpi=200)
    sns.set_theme(style="white")

    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=e_labels,
        yticklabels=t_labels,
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5
    )

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Evaluation / Test Generator", fontsize=11, fontweight="bold")
    plt.ylabel("Training Setting", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[SUCCESS] Heatmap saved to: {save_path}")
    else:
        plt.show()


# Alias
plot_generalization_heatmap = plot_logo_heatmap


def plot_robustness_curves(
    robustness_results: Dict[str, Dict[Any, Dict[str, float]]],
    save_path: Optional[str] = "./experiments/visualizations/robustness_curves.png"
):
    """
    Plot image degradation robustness curves (matching Figure 6).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    jpeg_results = robustness_results.get("jpeg", {})
    blur_results = robustness_results.get("blur", {})

    # 1. JPEG Compression Curve
    if jpeg_results:
        q_levels = sorted(list(jpeg_results.keys()))
        jpeg_accs = [jpeg_results[q].get("accuracy", 0.0) * 100 for q in q_levels]
        jpeg_aurocs = [jpeg_results[q].get("auroc", 0.5) * 100 for q in q_levels]

        axes[0].plot(q_levels, jpeg_accs, marker='o', linewidth=2.5, color='#e74c3c', label='LOTA (Acc %)')
        axes[0].plot(q_levels, jpeg_aurocs, marker='s', linewidth=2, linestyle='--', color='#2980b9', label='LOTA (AUROC %)')
        axes[0].set_title("Robustness to JPEG Compression", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("JPEG Quality Level", fontsize=10)
        axes[0].set_ylabel("Metric (%)", fontsize=10)
        axes[0].set_ylim([40, 105])
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc="lower left")

    # 2. Gaussian Blur Curve
    if blur_results:
        sigmas = sorted(list(blur_results.keys()))
        blur_accs = [blur_results[s].get("accuracy", 0.0) * 100 for s in sigmas]
        blur_aurocs = [blur_results[s].get("auroc", 0.5) * 100 for s in sigmas]

        axes[1].plot(sigmas, blur_accs, marker='o', linewidth=2.5, color='#e74c3c', label='LOTA (Acc %)')
        axes[1].plot(sigmas, blur_aurocs, marker='s', linewidth=2, linestyle='--', color='#2980b9', label='LOTA (AUROC %)')
        axes[1].set_title("Robustness to Gaussian Blur", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Gaussian Blur Sigma (radius)", fontsize=10)
        axes[1].set_ylabel("Metric (%)", fontsize=10)
        axes[1].set_ylim([40, 105])
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc="lower left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[SUCCESS] Robustness curves saved to: {save_path}")
    else:
        plt.show()
