# LOTA Codebase Audit — Iteration 1 & 2.1 Review & Gap Analysis

**Reference Standards**:
1. ICCV 2025 Paper: *LOTA: Bit-Planes Guided AI-Generated Image Detection* (Hongsong Wang et al.)
2. Official Repository: https://github.com/hongsong-wang/LOTA
3. Project Directives: `Things_to_improve.md` and `prompt_2.1.md`

---

## 1. Comprehensive Component Audit Table

| Component | Current Status | Correct / Incorrect | Specific Issue / Gap | Required Action |
| :--- | :--- | :--- | :--- | :--- |
| **Bit-Plane Decomposition (Eq. 1)** | Implemented in `src/forensic/bitplanes.py` | **Correct** | Extracts binary bit planes $x_k^c \in \{0, 1\}$ from 8-bit uint8 representations before normalization. | Retain and maintain unit tests with known integer verification. |
| **Low-Bit Composition (Eq. 2)** | Implemented in `src/forensic/bitplanes.py` | **Correct** | Composes $z^c = \sum_{k \in \mathcal{K}} 2^k x_k^c$ for $K=3$ ($k \in \{0, 1, 2\}$) with range $[0, 7]$. | Retain. Added synthetic verification in test suite. |
| **Noise Normalization (Eq. 3 & 4)** | Implemented in `src/forensic/normalization.py` | **Correct** | Implements both zero-centering thresholding ($\tilde{z} \in [-1, 1]$ or $[0, 255]$ scale) and scaling normalization. | Standardized thresholding as default matching paper Section 3.2. |
| **MGPS Gradient Convolutions (Eq. 5)** | Implemented in `src/forensic/mgps.py` | **Correct** | Implements 4 directional kernels ($D_1$ horizontal, $D_2$ vertical, $D_3$ 45°, $D_4$ 135°) with 2D conv. | Retain. Add explicit synthetic directional edge tests. |
| **MGPS Patch Scoring & Selection (Eq. 6)** | Implemented in `src/forensic/mgps.py` | **Correct** | Computes patch divergence score $g_p$ across non-overlapping $32 \times 32$ patches and selects $\arg\max_p g_p$. | Verified with argmax synthetic tests across $8 \times 8$ grid. |
| **Pretrained Backbone** | Implemented in `src/models/backbones.py` | **Correct** | Loads ImageNet pretrained ResNet-50 with replaced 1st conv layer for 3-channel noise patch and raw image. | Retain standard torchvision ResNet-50 weights. |
| **NBC Classifier** | Implemented in `src/models/nbc.py` | **Correct** | Feeds selected $32 \times 32 \times 3$ noise patch into ResNet-50 backbone followed by binary classification head. | Retain. Verified forward and backward gradients. |
| **NGC Classifier (Eq. 7)** | Implemented in `src/models/ngc.py` | **Approximation (Documented)** | Dual-stream architecture using cross-attention between raw image features and noise patch features. | Retain, but explicitly document as our architectural approximation in `docs/decisions.md`. Prioritize NBC first. |
| **Dataset Loading & Splitting** | Implemented in `src/data/` | **Correct with Safeguards** | Scans GenImage generator subdirectories, parses nature (real) and ai (fake), supports mock mode. | Added `scripts/check_data_integrity.py` to prevent split leakage. |
| **LOGO Split Partitioning** | Implemented in `src/data/splits.py` | **Correct** | Formats 4-generator representative splits (BigGAN, SD1.4, Midjourney, ADM) and strictly isolates held-out fake images. | Retain. Verified that held-out generator fake data is strictly excluded from training. |
| **Experiment Database Schema** | Implemented in `src/experiments/database.py` | **Updated (Verified)** | Stored experiments with explicit separation of `source_type` (`'experimental'`, `'published_reference'`, `'mock_fixture'`) and `is_mock`. | Maintained schema. Verified via unit and SQL queries. |
| **Reference Benchmarks Table** | Implemented in `src/experiments/database.py` | **Corrected (Iteration 2.1)** | Dedicated `reference_benchmarks` table with full paper traceability (`paper_table`, `paper_method`, `training_generator`, `evaluation_generator`). | Corrected ADM Table 1 (99.7%) and NGC Table 2 values (88.1%, 90.4%, 98.0%). Added `scripts/verify_reference_data.py`. |
| **Forensic Visualization & Fallback** | `scripts/visualize_forensic.py` & `notebooks/01_lota_mathematical_validation.ipynb` | **Updated (Iteration 2.1)** | Previous visualization silently substituted synthetic images when GenImage was absent. | **ACTION**: Added loud fallback warning, separated output filenames (`_synthetic.png` vs `_genimage_real_vs_fake.png`), blocked Figure 3 claims in synthetic mode, added readiness gate (`check_data_readiness.py`). |

---

## 2. Research Integrity Audit Decisions

1. **No Fabricated Results in Production Database**:
   - Local training scripts (`train.py`, `run_reproduction.py`, `run_logo.py`) will strictly log with `source_type='experimental', is_mock=0`.
   - Published figures from Table 1 and Table 2 of the ICCV 2025 paper are stored exclusively in `reference_benchmarks` tagged with `source_type='published_reference'`.
   - Test fixtures used to test the Streamlit demo or query routing in CI/mock mode are tagged with `source_type='mock_fixture', is_mock=1` and filtered out of research reports.

2. **Honest Compute & Performance Reporting**:
   - All experimental reports must state:
     - Dataset subset scale (e.g., 256 images per generator vs 200,000 in paper).
     - GPU hardware environment (Free Colab T4 / Kaggle P100 vs 8x A100 in paper).
     - Relative trends and generalization gaps rather than falsely claiming exact absolute metric reproduction.

---

## 3. Iteration 2.1 Audit & Corrections

1. **Reference Data Audit & Correction**:
   - **ADM Evaluation on SD1.5-Trained NBC (Table 1)**: Corrected transcription from 98.5% to **99.7%** (matching Table 1 of the paper).
   - **LOTA-ngc LOGO Evaluation (Table 2)**: Replaced erroneous values (BigGAN=100.0%, Midjourney=94.5%, ADM=99.1%) with verified figures from Table 2: **BigGAN = 88.1%**, **Midjourney = 90.4%**, **ADM = 98.0%**, **SD1.4 = 99.8%**.
   - Created [scripts/verify_reference_data.py](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/scripts/verify_reference_data.py) to validate and display traceability for all 16 reference benchmarks.

2. **Synthetic Fallback Transparency & Provenance Separation**:
   - Both [scripts/visualize_forensic.py](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/scripts/visualize_forensic.py) and [notebooks/01_lota_mathematical_validation.ipynb](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/notebooks/01_lota_mathematical_validation.ipynb) emit a prominent warning when the GenImage dataset is absent.
   - Output visual artifacts are saved with distinct filenames:
     - `experiments/visualizations/forensic_decomposition_synthetic.png`
     - `experiments/visualizations/forensic_decomposition_genimage_real_vs_fake.png`
   - Added [scripts/check_data_readiness.py](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/scripts/check_data_readiness.py) to gate real-data validation until legitimate GenImage image samples are available.
