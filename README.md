# LOTA-MP: Multi-Patch Evidence Aggregation for Bit-Plane Guided AI-Generated Image Detection

Course project for **DA 6021 — Deep Learning for Computer Vision** (Jul–Nov 2026).

| | |
|---|---|
| **Assigned paper** | Wang et al., *LOTA: Bit-Planes Guided AI-Generated Image Detection*, ICCV 2025 ([paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LOTA_Bit-Planes_Guided_AI-Generated_Image_Detection_ICCV_2025_paper.html)) |
| **Original implementation** | <https://github.com/hongsong-wang/LOTA> |
| **Team** | Viraj Chinta (230121066), Sundaram Pandey (230107082), Aditya Bhawsar (230101006), Abhyuday Kashyap (230122001), Sarvagya Tiwari (230122047) |

## 1. Project overview

**LOTA** detects AI-generated images from their *low-order bit-planes*. The three least significant bits of every pixel carry no visible content, only noise, and that noise differs between camera images and generated images. LOTA (i) builds a low-bit "noise image", (ii) cuts random 32×32 patches from it and keeps **the single patch** with the highest multi-directional gradient score (MGPS), and (iii) classifies that one patch with an ImageNet-pretrained ResNet-50.

**Our modification — LOTA-MP.** The baseline bases its whole decision on one 32×32 patch, about 1.5 % of a 256×256 image, and the paper's own ablation (Table 5) shows that the choice of patch moves BigGAN accuracy between 94.9 % and 69.9 %. LOTA-MP keeps the **K highest-gradient patches**, scores each with the *same* shared backbone, and averages their log-odds into one image-level decision, in training and at inference:

```
s̄(x) = (1/K) · Σ_k f_θ(z_(k))        p(fake | x) = σ(s̄(x))        L = BCE(s̄(x), y)
```

* Averaging log-odds is the naive-Bayes / product-of-experts fusion rule: for conditionally independent patches the sum of per-patch log-likelihood ratios is the optimal statistic.
* If each patch logit has variance σ² and pairwise correlation ρ, the fused logit has variance σ²·(ρ + (1−ρ)/K), so the decision margin grows by √(K / (1 + (K−1)ρ)).
* `K = 1` is exactly the original LOTA, **no parameters are added**, and the cost is K backbone passes instead of one.

Everything that differs from the baseline is in three places:

| File | What it contains |
|---|---|
| [`src/models/multi_patch.py`](src/models/multi_patch.py) | `MultiPatchLOTA` and the log-odds fusion rule (**the proposed module**) |
| [`src/forensic/mgps.py`](src/forensic/mgps.py) → `select_patches` | MGPS generalised from top-1 to top-K |
| [`src/training/losses.py`](src/training/losses.py) | loss on the fused logit |

The two configs, [`configs/baseline_lota.yaml`](configs/baseline_lota.yaml) and [`configs/proposed_lota_mp.yaml`](configs/proposed_lota_mp.yaml), differ in one line: `model.num_patches: 1` vs `4`.

## 2. Environment and dependencies

| | Version used for the reported results |
|---|---|
| Python | see [`results/environment.json`](results/environment.json) |
| PyTorch / torchvision / CUDA | see [`results/environment.json`](results/environment.json) |
| Hardware | one NVIDIA Tesla T4 (16 GB), 4 CPU cores (Kaggle Notebooks) |

Any Python ≥ 3.10 with PyTorch ≥ 2.0 works; a GPU with ≥ 12 GB memory is needed for the default batch size.

```bash
git clone https://github.com/SundaramPandey2005/LOTA-AI-Image-Forensics.git
cd LOTA-AI-Image-Forensics
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick the wheel for your CUDA
pip install -r requirements.txt
python -m pytest -q tests                                 # 30 unit tests, < 1 min on CPU
```

**No local GPU?** Open [`notebooks/run_experiments.ipynb`](notebooks/run_experiments.ipynb) on Kaggle or Colab; it runs every step below end to end.

## 3. Dataset preparation

We use **Tiny-GenImage** (<https://www.kaggle.com/datasets/yangsangtai/tiny-genimage>, 8.4 GB, CC BY-NC-SA 4.0), a subset of the [GenImage](https://github.com/GenImage-Dataset/GenImage) benchmark used in the paper. It keeps about 5,000 images for each of seven generators (BigGAN, Midjourney, Wukong, Stable Diffusion v1.5, ADM, GLIDE, VQDM) with the original GenImage folders and files. The full GenImage dataset has the same layout and works unchanged.

```bash
# Option A – download with kagglehub and verify
python scripts/prepare_data.py --download --data-root data/GenImage
# Option B – you already have (Tiny-)GenImage somewhere: just verify it
python scripts/prepare_data.py --data-root /path/to/GenImage
```

The script prints the image counts per generator and, on its last line, `DATA_ROOT=<path>`; pass that path as `--data-root` below. Expected layout:

```
<DATA_ROOT>/
├── imagenet_ai_0424_sdv5/          # also: imagenet_ai_0419_biggan, imagenet_midjourney,
│   ├── train/                      #       imagenet_ai_0424_wukong, imagenet_ai_0508_adm,
│   │   ├── ai/       *.png         #       imagenet_glide, imagenet_ai_0419_vqdm
│   │   └── nature/   *.JPEG
│   └── val/                        # used as the test set, as in the paper
│       ├── ai/
│       └── nature/
└── ...
```

**There is no offline pre-processing.** Bit-planes are extracted on the fly from the original files. Do not resize or re-encode the images: interpolation and re-compression rewrite exactly the low-order bits the method relies on. `train/` is split 90/10 (fixed, stratified) into training and held-out validation images for checkpoint selection; `val/` is only ever used for testing.

## 4. Training

Both models start from ImageNet-pretrained ResNet-50 weights (downloaded automatically by torchvision) and use the paper's optimiser settings: Adam, learning rate 1e-4 with polynomial decay, binary cross-entropy. We train for 10 epochs with 32 images per step (the paper: 30 epochs, batch 64, on the full-size subsets).

```bash
# Baseline: LOTA (one patch per image)
python scripts/train.py --config configs/baseline_lota.yaml    --data-root $DATA_ROOT --seed 0
# Proposed: LOTA-MP (K = 4 patches per image)
python scripts/train.py --config configs/proposed_lota_mp.yaml --data-root $DATA_ROOT --seed 0
```

Each run writes `outputs/runs/<name>_<generator>_s<seed>/` containing `config.yaml`, `train.log`, `history.json` and `best.pth`. Any setting can be overridden on the command line, e.g. `--set data.train_generator=biggan train.epochs=5 model.num_patches=8`.

To reproduce **all** reported experiments with one command (3 seeds, second training generator, robustness, ablations, latency; resumable):

```bash
python scripts/run_experiments.py --data-root $DATA_ROOT --out outputs
python scripts/run_experiments.py --data-root $DATA_ROOT --out outputs_quick --quick   # 2-minute smoke test
```

## 5. Inference and evaluation

```bash
# Test a checkpoint on every generator; prints ACC for K = 1, 2, 4, 8 fused patches
python scripts/evaluate.py --checkpoint outputs/runs/lota_sdv5_s0/best.pth    --data-root $DATA_ROOT
python scripts/evaluate.py --checkpoint outputs/runs/lota_mp_sdv5_s0/best.pth --data-root $DATA_ROOT

# Optional: unseen degradations (as in Fig. 6 of the paper) and latency / parameter count
python scripts/evaluate.py --checkpoint outputs/runs/lota_mp_sdv5_s0/best.pth --data-root $DATA_ROOT \
       --perturbation clean jpeg90 blur1 --latency

# Build the result tables from all runs
python scripts/summarize_results.py --out outputs
```

* **Input:** a checkpoint (`best.pth`) and the dataset root. **Output:** `eval_<perturbation>.json` (ACC / AP / AUROC per generator and averaged), `logits_<perturbation>.npz` (raw per-patch logits) and `eval.log`, written next to the checkpoint; `summarize_results.py` writes `outputs/summary.md` and `summary.json`.
* **Metrics** follow the paper: accuracy at threshold 0.5 and average precision, per test generator and averaged over generators. "AI-generated" is the positive class.
* **Reading the output.** The baseline number is the `lota_*` checkpoint at `K1`; the proposed number is the `lota_mp_*` checkpoint at `K4`. Test-time patches are deterministic per image, so both models are scored on identical patches and repeated runs give identical numbers.

## 6. Checkpoints and results

Results, logs and weights will be added here once the final run completes.

## 7. Code organisation and reproducibility

```
configs/
  baseline_lota.yaml          LOTA, K = 1
  proposed_lota_mp.yaml       LOTA-MP, K = 4 (only model.num_patches differs)
src/
  forensic/lowbit.py          low-bit noise image (Eq. 1–4), native resolution
  forensic/mgps.py            gradient score (Eq. 5) and top-K patch selection   ← modified
  data/genimage.py            GenImage discovery, splits, on-the-fly patch extraction
  models/patch_classifier.py  baseline noise-based classifier (ResNet-50)
  models/multi_patch.py       LOTA-MP fusion module                              ← proposed
  training/losses.py          loss on the fused logit                            ← proposed
  training/engine.py          training loop, checkpointing
  evaluation/evaluator.py     cross-generator evaluation, latency benchmark
scripts/
  prepare_data.py  train.py  evaluate.py  run_experiments.py  summarize_results.py
notebooks/run_experiments.ipynb   one-click Kaggle / Colab reproduction
tests/                            unit tests (bit-planes, MGPS, fusion maths, train/reload)
report/                           one-page course report (LaTeX)
results/                          metrics, logs and tables of the reported run
```

Reproducibility notes:

* **Faithful baseline.** Following the official loader, bit-planes are taken from the image at its original size, 64 random 32×32 crops are scored, and the selected patch is upsampled to 256×256. We use the *scaling* normalisation, the variant implemented in the official release (the paper reports scaling and thresholding within 0.2 points of each other); `forensic.bit_mode=thresholding` selects Eq. (4).
* **Fair comparison.** Baseline and proposed runs share data, split, seed, optimiser, schedule and number of images per step; only `model.num_patches` changes.
* **Seeds.** `--seed` fixes model initialisation, data order and training crops. Test crops depend only on the file name.
* **No test-set tuning.** Checkpoints are selected on images held out from `train/`.

## Citation

```bibtex
@inproceedings{wang2025lota,
  title     = {LOTA: Bit-Planes Guided AI-Generated Image Detection},
  author    = {Wang, Hongsong and Cheng, Renxi and Zhang, Yang and Han, Chaolei and Gui, Jie},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
