"""
Forensic Visualization Script for LOTA
Demonstrates bit-plane slicing, noise composition, thresholding, and MGPS patch selection.
Strictly separates SYNTHETIC mathematical validation from REAL GenImage forensic validation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.forensic.bitplanes import extract_bit_planes, compose_low_bit_planes
from src.forensic.normalization import normalize_noise_thresholding, normalize_noise_scaling
from src.forensic.mgps import maximum_gradient_patch_selection, compute_patch_divergence_scores
from src.data.dataset import GenImageDataset


def create_synthetic_forensic_sample(size: int = 256) -> np.ndarray:
    """Create a synthetic test image with natural gradients + high-frequency noise regions."""
    np.random.seed(42)
    x = np.linspace(0, 255, size, dtype=np.float32)
    y = np.linspace(0, 255, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = (0.5 * xx + 0.5 * yy).astype(np.uint8)
    
    img = np.stack([base, base, base], axis=-1)
    
    # Inject high-frequency low-bit artifact in a specific 32x32 patch region (row 2, col 5)
    artifact = (np.random.randint(0, 8, (32, 32, 3))).astype(np.uint8)
    img[64:96, 160:192] = (img[64:96, 160:192] & np.uint8(248)) | artifact
    return img


def run_forensic_visualization(
    image_path: str = None,
    output_path: str = None,
    dataset_root: str = "./data/GenImage"
):
    # Determine sample source and provenance
    is_real_genimage = False
    source_label = "SYNTHETIC"

    if image_path and os.path.exists(image_path):
        raw_pil = Image.open(image_path).convert("RGB").resize((256, 256))
        raw_np = np.array(raw_pil)
        source_label = f"EXTERNAL IMAGE ({os.path.basename(image_path)})"
    else:
        # Check if local GenImage dataset exists
        ds = GenImageDataset(root_dir=dataset_root, split="val", use_mock_data=False)
        if not ds.use_mock_data and len(ds.samples) > 0:
            sample_item = ds[0]
            raw_np = sample_item["raw_image"].numpy().transpose(1, 2, 0).astype(np.uint8)
            is_real_genimage = True
            source_label = f"GENIMAGE REAL DATASET ({sample_item.get('generator', 'unknown')})"
        else:
            # Synthetic Fallback with LOUD warning
            print("\n" + "*" * 80)
            print("  [WARNING] GenImage dataset was not found at expected path: data/GenImage.")
            print("  This script is running in SYNTHETIC MATHEMATICAL VALIDATION MODE.")
            print("  The generated visualization validates mathematical pipeline mechanics only.")
            print("  It MUST NOT be interpreted as evidence of real-image or AI-generated image forensic behavior.")
            print("  Qualitative comparisons with Figure 3 of the LOTA paper are DISABLED until real GenImage samples are available.")
            print("*" * 80 + "\n")
            raw_np = create_synthetic_forensic_sample(size=256)
            source_label = "SYNTHETIC MATHEMATICAL VALIDATION SAMPLE"

    # Set default provenance-aware output path if not specified
    if output_path is None:
        if is_real_genimage:
            output_path = "./experiments/visualizations/forensic_decomposition_genimage_real_vs_fake.png"
        else:
            output_path = "./experiments/visualizations/forensic_decomposition_synthetic.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_tensor = torch.from_numpy(raw_np).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, 256, 256)
    
    # 1. Extract all 8 bit planes
    bit_planes = extract_bit_planes(raw_np, bits=list(range(8))) # (256, 256, 3, 8)
    
    # 2. Compose low 3 bit planes (k=0, 1, 2)
    z = compose_low_bit_planes(raw_tensor, bit_indices=[0, 1, 2]) # (1, 3, 256, 256)
    
    # 3. Apply Thresholding (Eq. 4) and Scaling (Eq. 3)
    z_thresh = normalize_noise_thresholding(z) # (1, 3, 256, 256)
    z_scaled = normalize_noise_scaling(z)     # (1, 3, 256, 256)
    
    # 4. MGPS Patch Selection
    selected_patch, best_idx = maximum_gradient_patch_selection(
        z_thresh, patch_size=32, strategy="max_gradient"
    )
    
    # Compute divergence score heatmap across all 8x8 patches
    num_h = 256 // 32
    num_w = 256 // 32
    patches_unfolded = z_thresh.view(1, 3, num_h, 32, num_w, 32).permute(0, 2, 4, 1, 3, 5).contiguous().view(1, num_h * num_w, 3, 32, 32)
    scores = compute_patch_divergence_scores(patches_unfolded).view(num_h, num_w).cpu().numpy()
    
    # Calculate bounding box of selected patch
    idx = best_idx.item()
    best_row = idx // num_w
    best_col = idx % num_w
    bbox_x, bbox_y = best_col * 32, best_row * 32
    
    # Plotting complete forensic pipeline
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Original Image, Composed Noise, Thresholded, Scaled, MGPS Heatmap, Selected Patch
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(raw_np)
    ax1.set_title(f"Input: [{source_label}]")
    ax1.axis("off")
    
    ax2 = plt.subplot(3, 4, 2)
    z_disp = z[0].permute(1, 2, 0).cpu().numpy() / 7.0
    ax2.imshow(z_disp)
    ax2.set_title("Composed Low-Bits ($z^c$, $K=3$)")
    ax2.axis("off")
    
    ax3 = plt.subplot(3, 4, 3)
    z_thresh_disp = z_thresh[0].permute(1, 2, 0).cpu().numpy() / 255.0
    ax3.imshow(z_thresh_disp)
    rect = patches.Rectangle((bbox_x, bbox_y), 32, 32, linewidth=2.5, edgecolor="red", facecolor="none")
    ax3.add_patch(rect)
    ax3.set_title(f"Thresholded Noise (z_tilde) [MGPS: #{idx}]")
    ax3.axis("off")
    
    ax4 = plt.subplot(3, 4, 4)
    im_heat = ax4.imshow(scores, cmap="hot", interpolation="nearest")
    plt.colorbar(im_heat, ax=ax4, fraction=0.046, pad=0.04)
    rect_heat = patches.Rectangle((best_col - 0.5, best_row - 0.5), 1, 1, linewidth=2.5, edgecolor="cyan", facecolor="none")
    ax4.add_patch(rect_heat)
    ax4.set_title("MGPS Divergence Heatmap (g_p)")
    ax4.axis("off")
    
    # Row 2 & 3: 8 Individual Bit Planes (k=0..7)
    for k in range(8):
        ax_bit = plt.subplot(3, 4, 5 + k)
        bit_img = bit_planes[..., 1, k] # Green channel
        ax_bit.imshow(bit_img, cmap="gray")
        is_low = " (LOTA Low-Bit)" if k < 3 else ""
        ax_bit.set_title(f"Bit-Plane {k} ($2^{k}$){is_low}", color="red" if k < 3 else "black")
        ax_bit.axis("off")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Forensic visualization saved to: {output_path} (Provenance: {source_label})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LOTA Bit-Plane Forensic Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (optional)")
    parser.add_argument("--dataset_root", type=str, default="./data/GenImage", help="Dataset root directory")
    args = parser.parse_args()
    
    run_forensic_visualization(args.image, args.output, args.dataset_root)
