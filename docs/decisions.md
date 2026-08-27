# LOTA Architectural & Experimental Decisions Log

This document records all key decisions made during the implementation and reproduction of the LOTA (ICCV 2025) research pipeline.

Hierarchy of Authority:
- **Priority 1 — Research Paper**: *LOTA: Bit-Planes Guided AI-Generated Image Detection* (ICCV 2025)
- **Priority 2 — Official Repository**: https://github.com/hongsong-wang/LOTA (used to clarify underspecified details)
- **Priority 3 — Our Experimental Decision**: Choices made to accommodate compute constraints, testability, or software engineering discipline.

---

## 1. Bit-Plane Slicing and Integer Handling

- **Context**: Bit extraction requires exact binary plane decomposition. Applying floating-point transforms or normalizing before bit slicing would destroy discrete binary planes.
- **Options Considered**:
  - Option A: Normalize image to $[0, 1]$ floats and attempt bitwise operations (Incorrect).
  - Option B: Maintain unnormalized uint8 or integer tensor in $[0, 255]$ for bit slicing, then normalize (Correct).
- **Decision**: Option B.
- **Reason**: Bit-plane slicing $x^c = \sum_{k=0}^7 2^k x_k^c$ operates strictly on 8-bit integer channel values.
- **Evidence**: Paper Section 3.1, Equation 1.
- **Impact**: Guarantees mathematically exact lossless recovery $\sum_{k=0}^7 2^k x_k^c \equiv x^c$.

---

## 2. Image Resizing & Interpolation Mode

- **Context**: Input images must be standardized to $256 \times 256$ before forensic analysis. The interpolation mode influences high-frequency pixel transitions.
- **Options Considered**:
  - Option A: Nearest Neighbor interpolation (preserves exact values but creates blocky aliasing).
  - Option B: Bilinear interpolation (smooth continuous resampling).
  - Option C: Bicubic interpolation.
- **Decision**: Option B (Bilinear Interpolation).
- **Reason**: The paper does not specify the exact interpolation mode in text; the official LOTA repository uses `PIL.Image.BILINEAR` / `torchvision.transforms.InterpolationMode.BILINEAR`.
- **Evidence**: Official LOTA repository (`dataset.py`).
- **Impact**: Consistent with official data ingestion pipeline.

---

## 3. Noise Normalization: Zero-Centering Thresholding vs Scaling

- **Context**: Composed low 3-bit noise maps $z^c \in [0, 7]$ contain high frequency artifacts. The paper defines two normalization techniques: scaling (Eq. 3) and zero-centering thresholding (Eq. 4).
- **Options Considered**:
  - Option A: Linear scaling to $[0, 255]$ ($z^c \times \frac{255}{7}$).
  - Option B: Zero-centering thresholding mapping values $> 3.5$ to positive and $\le 3.5$ to negative/zero-centered range.
- **Decision**: Zero-centering Thresholding (Eq. 4) as primary default, with scaling supported for ablation studies.
- **Reason**: Thresholding accentuates subtle phase transitions in low-order bit planes and yields superior empirical detection in the paper's ablation studies (Section 4.4, Table 3).
- **Evidence**: Paper Section 3.2, Equation 4.
- **Impact**: Provides faithful reproduction of the paper's primary baseline.

---

## 4. Maximum Gradient Patch Selection (MGPS) Convolutions

- **Context**: MGPS extracts a $32 \times 32$ patch with the highest gradient divergence from the normalized noise map $\tilde{z}$ using 4 directional kernels ($D_1, D_2, D_3, D_4$).
- **Options Considered**:
  - Option A: Standard Sobel 2-directional gradient (Horizontal/Vertical only).
  - Option B: 4-directional gradient kernels ($D_1=[1, -1], D_2=[1; -1], D_3=[1, 0; 0, -1], D_4=[0, 1; -1, 0]$).
- **Decision**: Option B.
- **Reason**: AI generation artifacts frequently exhibit diagonal high-frequency lattice noise in addition to axial gradients.
- **Evidence**: Paper Section 3.3, Equations 5 and 6.
- **Impact**: Captures 45° and 135° diagonal noise patterns essential for synthetic artifact localization.

---

## 5. Noise-Based Classifier (NBC) Backbone & Weights

- **Context**: The classifier network processes selected $32 \times 32 \times 3$ patches to predict real vs fake probabilities.
- **Options Considered**:
  - Option A: Train a ResNet-50 from scratch.
  - Option B: Initialize standard ImageNet-1k pretrained ResNet-50 (`weights=ResNet50_Weights.IMAGENET1K_V2`), adapt first convolution layer for $32 \times 32$ inputs, and fine-tune with Adam.
- **Decision**: Option B.
- **Reason**: Follows the paper's methodology of transfer learning from ImageNet-pretrained weights with cosine learning rate scheduling.
- **Evidence**: Paper Section 4.1 (Implementation Details).
- **Impact**: Rapid convergence within 10–15 epochs.

---

## 6. Noise-Guided Classifier (NGC) Dual-Stream Cross-Attention

- **Context**: NGC augments the NBC pipeline with raw RGB image features using cross-attention (Eq. 7).
- **Options Considered**:
  - Option A: Claim exact black-box reproduction of NGC without full layer weights.
  - Option B: Implement an explicit dual-stream architecture with ResNet-50 feature maps, spatial projection, and multi-head cross-attention ($Q=\text{Raw}, K,V=\text{Noise}$), clearly documenting it as an architectural implementation decision.
- **Decision**: Option B.
- **Reason**: Research integrity requires explicitly distinguishing faithful paper baseline (NBC) from our dual-stream cross-attention approximation (NGC).
- **Evidence**: Paper Section 3.4, Equation 7.
- **Impact**: Transparent defensibility during technical interviews.

---

## 7. Data Provenance Separation in Experiment Database

- **Context**: The SQLite experiment database previously lacked strict metadata separating published paper metrics, local GPU training runs, and mock test fixtures.
- **Options Considered**:
  - Option A: Keep a single unstructured table where paper numbers and local runs share IDs.
  - Option B: Enforce `source_type` (`'experimental'`, `'published_reference'`, `'mock_fixture'`) and `is_mock` boolean across all database tables (`experiments`, `metrics`, `robustness_results`), ensuring production queries exclude mock fixtures by default.
- **Decision**: Option B.
- **Reason**: Non-negotiable research integrity requirement to prevent accidental conflation of published figures with local measurements.
- **Evidence**: `Things_to_improve.md` Addendum Section 1.
- **Impact**: Guarantees zero data fabrication or ambiguous provenance.

---

## 8. Representative 4-Generator LOGO Scope

- **Context**: Full 8-generator LOGO requires $8 \times 30$ training runs, which exceeds free Colab/Kaggle compute limits.
- **Options Considered**:
  - Option A: Train only on 1 generator and make speculative cross-generator claims.
  - Option B: Execute 4 representative LOGO rotations (BigGAN, SD1.4, Midjourney, ADM) covering GAN, Latent Diffusion, Proprietary Diffusion, and Ablated Diffusion archetypes, evaluating each trained model across all 8 generator test sets.
- **Decision**: Option B.
- **Reason**: Provides maximum family diversity across generative paradigms while maintaining feasible compute budgets ($4 \times 10$ epochs).
- **Evidence**: Project Directive Section 12.
- **Impact**: Defensible generalization benchmark with statistical validity.

---

## 9. Published Reference Benchmark Corrections (Iteration 2.1)

- **Context**: Audit of `scripts/seed_reference_data.py` revealed transcription errors against the published ICCV 2025 LOTA paper:
  1. *Table 1 (LOTA-nbc, SD1.5-trained row)*: The ADM evaluation accuracy was previously transcribed as 98.5%. The actual verified value from Paper Table 1 is **99.7%**.
  2. *Table 2 (LOTA-ngc, LOGO generalization row)*: Values were previously seeded with incorrect numbers (BigGAN=100.0%, Midjourney=94.5%, ADM=99.1%). The actual verified values from Paper Table 2 are **BigGAN = 88.1%**, **Midjourney = 90.4%**, **ADM = 98.0%**, **SD1.4 = 99.8%**.
- **Decision**: Corrected all 16 reference values to match the exact published paper figures, added complete provenance metadata (`paper_table`, `paper_method`, `training_generator`, `evaluation_generator`), and created [scripts/verify_reference_data.py](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/scripts/verify_reference_data.py) for verification.
- **Impact**: 100% verified traceability of all reference benchmark data.

---

## 10. Synthetic Fallback Transparency & Provenance-Aware Artifacts (Iteration 2.1)

- **Context**: When the real GenImage dataset is absent locally, forensic visualization previously fell back to synthetic samples without loud warnings, leading to potential confusion between synthetic mathematical testing and real forensic analysis.
- **Decision**:
  1. **Loud Prominent Warning**: Added a prominent warning box in [scripts/visualize_forensic.py](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/scripts/visualize_forensic.py) and [notebooks/01_lota_mathematical_validation.ipynb](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/notebooks/01_lota_mathematical_validation.ipynb).
  2. **Separated Artifact Filenames**: Output files are saved distinctly as `forensic_decomposition_synthetic.png` (Mode A) and `forensic_decomposition_genimage_real_vs_fake.png` (Mode B).
  3. **Real-Data Readiness Gate**: Implemented [scripts/check_data_readiness.py](file:///g:/Antigravity%20Projects/DLCV%20LOTA%20Project/scripts/check_data_readiness.py) to gate real-data validation until a genuine GenImage subset is loaded.
  4. **Blocked Claims**: Qualitative comparisons to Figure 3 of the paper are disabled during synthetic validation mode.

---

## 11. Strict Mock-Data Policy & Dual-Gate Verification (Iteration 2.2)

- **Context**: Silent fallback to mock data inside `src/data/dataset.py` allowed missing directories to pass unnoticed during development, risking false claims of real experimental execution.
- **Decision**:
  - **Elimination of Silent Fallbacks**: `GenImageDataset` raises explicit `FileNotFoundError` or `RuntimeError` when `use_mock_data=False` and dataset paths are missing.
  - **Explicit Gate Separation**:
    - **Gate A (Infrastructure Gate)**: Tests software infrastructure using synthetic mocks (`scripts/run_infrastructure_check.py`) and reports `INFRASTRUCTURE GATE: PASSED`.
    - **Gate B (Real-Data Pilot Gate)**: Tests empirical pipeline on real GenImage images (`scripts/run_real_data_pilot.py`) and reports `REAL-DATA PILOT GATE: PASSED` or `NOT READY`.
- **Impact**: Completely eliminates accidental execution of mock training masquerading as real data.

---

## 12. LOGO 4-Generator to 3-Generator Compute Fallback Contingency (Iteration 2.2)

- **Context**: If GPU limits or dataset download bottlenecks on Google Colab/Kaggle prevent training on all 4 representative generators (BigGAN, SD1.4, Midjourney, ADM).
- **Decision**:
  - **Primary Plan**: 4 representative generators (BigGAN, SD1.4, Midjourney, ADM) covering GAN, Latent Diffusion, Commercial Diffusion, and Pixel-Space Diffusion.
  - **Fallback Contingency**: If compute budget is severely constrained, scale down to 3 representative generators:
    1. **BigGAN** (GAN archetype)
    2. **SD v1.4** (Latent Diffusion archetype)
    3. **Midjourney** (Commercial Diffusion archetype)
  - **Rule**: Retain multi-family diversity (GAN vs LDM vs Proprietary) and document the exact compute justification in the final research report. Never silently shrink datasets below statistically significant sample counts ($N \ge 1000$).
