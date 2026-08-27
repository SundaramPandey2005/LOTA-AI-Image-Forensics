# LOTA Empirical Experiment Protocol & Compute Strategy

## 1. Dataset Architecture & Leakage Safeguards

### Dataset Source Structure (GenImage)
The benchmark uses the **GenImage** dataset comprising real nature images (ImageNet-1k validation/train subset) paired with AI-generated counter-parts from 8 generative models:

1. **BigGAN**: High-capacity Generative Adversarial Network (GAN archetype)
2. **SD v1.4**: CompVis Latent Diffusion Model (LDM archetype)
3. **SD v1.5**: RunwayML Latent Diffusion Model
4. **Midjourney**: Commercial high-fidelity proprietary diffusion (Stress case)
5. **ADM**: Ablated Diffusion Model (Pixel-space diffusion archetype)
6. **GLIDE**: OpenAI Guided Diffusion Model
7. **Wukong**: Chinese text-to-image diffusion model
8. **VQDM**: Vector Quantized Diffusion Model

### Leakage Safeguards
- **Real Image Shared vs Isolated Analysis**: In GenImage, real images are drawn from ImageNet. When splitting datasets for LOGO experiments:
  - **Rule 1 (Fake Data Isolation)**: Held-out generator fake images are strictly excluded from the training split.
  - **Rule 2 (Split Separation)**: No image filenames from the test set may appear in the training or validation splits.
  - **Rule 3 (Automated Integrity Audit)**: `scripts/check_data_integrity.py` must be executed before training to assert zero intersection between train and test sample sets.

---

## 2. Compute Cost Classification

| Category | Description | Examples | Compute Cost | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **Category A** | **Cheap / High Reward** | Unit tests, synthetic validation, forensic visual decomposition, post-training robustness degradation evaluation (JPEG/Blur), cross-generator test matrix inference on trained models, SQL queries. | CPU / < 2 min GPU | **Highest** (Always run first) |
| **Category B** | **Moderate Cost / High Research Value** | Single-generator SD v1.5 reproduction (10 epochs), 4 representative LOGO rotations (BigGAN, SD1.4, Midjourney, ADM at 10 epochs each), Bit-plane depth & MGPS patch ablations. | ~15–30 min per run on Free Colab T4 / Kaggle P100 | **Core Research Focus** |
| **Category C** | **Expensive / Optional** | Full 8-generator LOGO rotations, extensive hyperparameter sweeps, ViT transformer backbones, adversarial FGSM/PGD attacks. | > 5 hours GPU | **Excluded / Future Work** |

---

## 3. Staged Development & Verification Lifecycle

```
Stage A: Development Scale (Local CPU / Mock Data)
  └── Verify code execution, bit-plane recovery, tensor shapes, gradient flow, DB logging.
            ↓
Stage B: Pilot Empirical Scale (Single Small Run on Target Hardware)
  └── Run pilot check (e.g. 128 train / 64 val samples for 3 epochs).
  └── Measure: VRAM peak, epoch duration, image throughput (img/sec), checkpoint validity.
            ↓
Stage C: Locked Production Scale
  └── Lock exact sample size per generator (e.g. 2,000–5,000 per class), batch size (64), epochs (10–15).
  └── Execute 4 LOGO rotations and log verified metrics to SQLite.
```

---

## 4. Empirical Pilot Measurements (Recorded on Baseline Environment)

- **Environment**: Python 3.12, PyTorch 2.13.0, Torchvision 0.28.0 (CPU mode)
- **Pilot Configuration**:
  - Training Samples: 128 (64 real, 64 fake)
  - Validation Samples: 64 (32 real, 32 fake)
  - Batch Size: 32
  - Epochs: 3
- **Measured Metrics**:
  - Processed: 384 images in 67.95s
  - Throughput: 5.65 images/sec (CPU)
  - Checkpoint generated: `./checkpoints/pilot_nbc_best.pth`
  - Pilot Run ID: `EXP_PILOT_VERIFICATION` (logged with `source_type='experimental', is_mock=0`)

---

## 5. Contingency Protocols & Fallback Scenarios

1. **Colab / Kaggle Disconnection**:
   - Checkpoints saved per epoch with optimizer state and best validation AUROC score.
   - Script resumes seamlessly from `--resume checkpoints/latest.pth`.
2. **GPU Out of Memory (OOM)**:
   - Dynamic batch size halving ($64 \rightarrow 32 \rightarrow 16$) with gradient accumulation.
   - Mixed Precision (`torch.cuda.amp.autocast()`) active by default on CUDA.
3. **Dataset Download Constraints**:
   - For rapid initial experimentation, download the minimal single-generator subset (SD v1.5 val split) containing 100 real and 100 fake images.

---

## 6. Dataset Acquisition & Local Setup Guide (Iteration 2.1)

### Official GenImage Source
- **Official GenImage Repository**: https://github.com/GenImage-Dataset/GenImage
- **Hugging Face Mirror**: https://huggingface.co/datasets/GenImage
- **OpenDataLab**: https://opendatalab.com/GenImage

### Recommended Minimal Validation Subset
For initial forensic validation and sanity testing without downloading the entire 100GB+ dataset:
1. Download only the **Stable Diffusion v1.5** generator subset (or BigGAN).
2. Extract the directory structure into `./data/GenImage/` as follows:
```
data/GenImage/
└── sd15/
    ├── train/
    │   ├── nature/     # Real ImageNet photos (e.g. 0.png, 1.png, ...)
    │   └── ai/         # AI-generated images (e.g. 0.png, 1.png, ...)
    └── val/
        ├── nature/     # Real validation photos
        └── ai/         # AI validation photos
```

### Verification
Run the real-data readiness gate:
```powershell
.venv\Scripts\python.exe scripts/check_data_readiness.py
```
When valid images are detected, the status transitions to:
```
REAL-DATA VALIDATION STATUS: READY
```
This unlocks Mode B in [notebooks/01_lota_mathematical_validation.ipynb](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/notebooks/01_lota_mathematical_validation.ipynb) and generates `forensic_decomposition_genimage_real_vs_fake.png`.
