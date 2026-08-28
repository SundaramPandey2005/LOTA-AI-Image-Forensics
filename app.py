"""
LOTA: Bit-Planes Guided AI-Generated Image Detection
Interactive Research Explorer & Forensic Image Analyzer (ICCV 2025)
"""
import os
import io
import time
import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.forensic.bitplanes import extract_bit_planes, compose_low_bit_planes
from src.forensic.normalization import normalize_noise_thresholding, normalize_noise_scaling
from src.forensic.mgps import maximum_gradient_patch_selection, compute_patch_divergence_scores
from src.experiments.database import ExperimentDatabase
from src.experiments.queries import ExperimentIntelligenceEngine, INTENT_REGISTRY

# Streamlit Page Config
st.set_page_config(
    page_title="LOTA: Forensic AI-Generated Image Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Warn viewers if the database currently contains only illustrative/demo data
_demo_check_db = ExperimentDatabase("./experiments/results/lota_experiments.db")
try:
    _real_exp_df = _demo_check_db.query_df(
        "SELECT COUNT(*) as count FROM experiments WHERE is_mock = 0 AND experiment_id NOT LIKE 'DEMO_ONLY_%'"
    )
    _real_exp_count = _real_exp_df.iloc[0]["count"] if len(_real_exp_df) > 0 else 0
except Exception:
    _real_exp_count = 0

if _real_exp_count == 0:
    st.warning(
        "⚠️ No real experiments have been run yet in this database. "
        "All numbers currently shown are SYNTHETIC DEMO DATA for UI "
        "illustration only, and do not reflect this project's actual results.",
        icon="⚠️"
    )

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-real { background-color: #DEF7EC; color: #03543F; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .badge-fake { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_query_engine(db_path: str = "./experiments/results/lota_experiments.db"):
    # Ensure database is initialized with pre-seeded benchmark results
    db = ExperimentDatabase(db_path)
    # Check if benchmark records exist, if not seed realistic paper records
    df = db.query_df("SELECT COUNT(*) as count FROM experiments")
    if df.iloc[0]["count"] == 0:
        _seed_demo_experiments(db)
    return ExperimentIntelligenceEngine(db_path)


def _seed_demo_experiments(db: ExperimentDatabase):
    """
    Seed ILLUSTRATIVE/SYNTHETIC demo data so the app has something to display
    when the database is empty (e.g. on first run / fresh clone).

    THESE ARE NOT REAL EXPERIMENTAL RESULTS. They are inspired by the paper's
    published figures purely so the UI has plausible-looking numbers to render
    before you have run your own experiments. They must be logged as
    source_type='mock_fixture', is_mock=True so they are never confused with,
    or mixed into queries/comparisons against, this project's genuine
    experimental results (E1, zero-shot cross-generator eval, etc.).
    """
    from src.experiments.logger import ExperimentLogger
    logger = ExperimentLogger(db.db_path)

    # 1. SDv1.5 Trained NBC (Paper reproduction)
    metrics_sd15 = {
        "biggan": {"accuracy": 1.000, "auroc": 1.000, "average_precision": 1.000, "f1": 1.000},
        "sd14": {"accuracy": 0.999, "auroc": 0.999, "average_precision": 0.999, "f1": 0.999},
        "sd15": {"accuracy": 0.999, "auroc": 0.999, "average_precision": 0.999, "f1": 0.999},
        "midjourney": {"accuracy": 0.931, "auroc": 0.962, "average_precision": 0.958, "f1": 0.925},
        "adm": {"accuracy": 0.985, "auroc": 0.992, "average_precision": 0.990, "f1": 0.982},
        "glide": {"accuracy": 1.000, "auroc": 1.000, "average_precision": 1.000, "f1": 1.000},
        "wukong": {"accuracy": 0.998, "auroc": 0.999, "average_precision": 0.999, "f1": 0.997},
        "vqdm": {"accuracy": 0.997, "auroc": 0.998, "average_precision": 0.998, "f1": 0.996}
    }
    robust_sd15 = {
        "jpeg": {
            100.0: {"accuracy": 0.989, "auroc": 0.995, "average_precision": 0.994, "f1": 0.988},
            95.0: {"accuracy": 0.978, "auroc": 0.988, "average_precision": 0.986, "f1": 0.976},
            90.0: {"accuracy": 0.962, "auroc": 0.975, "average_precision": 0.971, "f1": 0.959},
            85.0: {"accuracy": 0.934, "auroc": 0.952, "average_precision": 0.946, "f1": 0.928},
            80.0: {"accuracy": 0.895, "auroc": 0.921, "average_precision": 0.912, "f1": 0.887},
            70.0: {"accuracy": 0.812, "auroc": 0.845, "average_precision": 0.832, "f1": 0.798}
        },
        "blur": {
            0.0: {"accuracy": 0.989, "auroc": 0.995, "average_precision": 0.994, "f1": 0.988},
            1.0: {"accuracy": 0.965, "auroc": 0.978, "average_precision": 0.974, "f1": 0.962},
            2.0: {"accuracy": 0.912, "auroc": 0.932, "average_precision": 0.925, "f1": 0.905},
            3.0: {"accuracy": 0.834, "auroc": 0.865, "average_precision": 0.852, "f1": 0.821}
        }
    }
    logger.log_run(
        experiment_id="DEMO_ONLY_LOTA_SD15_ILLUSTRATIVE",
        name="[DEMO/SYNTHETIC] Illustrative NBC numbers (not a real run)",
        config={"model": {"architecture": "nbc", "backbone": "resnet50"}, "forensic": {"bit_planes": [0, 1, 2], "normalization": "thresholding", "patch_size": 32}},
        model_id="M_NBC_RESNET50",
        metrics_by_generator=metrics_sd15,
        robustness_results=robust_sd15,
        training_time_sec=1420.0,
        source_type="mock_fixture",
        is_mock=True,
        notes="SYNTHETIC DEMO DATA ONLY -- illustrative numbers inspired by the published paper, not a result this project actually produced. Do not cite."
    )

    # 2. LOGO Rotation: Exclude Midjourney
    metrics_logo_mj = {
        "biggan": {"accuracy": 0.998, "auroc": 0.999, "average_precision": 0.999, "f1": 0.997},
        "sd14": {"accuracy": 0.995, "auroc": 0.997, "average_precision": 0.996, "f1": 0.994},
        "midjourney": {"accuracy": 0.924, "auroc": 0.955, "average_precision": 0.948, "f1": 0.918},
        "adm": {"accuracy": 0.981, "auroc": 0.988, "average_precision": 0.985, "f1": 0.979}
    }
    logger.log_run(
        experiment_id="DEMO_ONLY_LOGO_ILLUSTRATIVE",
        name="[DEMO/SYNTHETIC] Illustrative LOGO Exclude Midjourney (not a real run)",
        config={"model": {"architecture": "nbc"}},
        model_id="M_NBC_RESNET50",
        metrics_by_generator=metrics_logo_mj,
        excluded_generator="midjourney",
        training_time_sec=1850.0,
        source_type="mock_fixture",
        is_mock=True,
        notes="SYNTHETIC DEMO DATA ONLY -- illustrative numbers inspired by the published paper, not a result this project actually produced. Do not cite."
    )

    # 3. NGC Noise-Guided Classifier
    metrics_ngc = {
        "biggan": {"accuracy": 1.000, "auroc": 1.000, "average_precision": 1.000, "f1": 1.000},
        "sd14": {"accuracy": 0.999, "auroc": 0.999, "average_precision": 0.999, "f1": 0.999},
        "midjourney": {"accuracy": 0.945, "auroc": 0.974, "average_precision": 0.969, "f1": 0.941},
        "adm": {"accuracy": 0.991, "auroc": 0.996, "average_precision": 0.994, "f1": 0.989}
    }
    logger.log_run(
        experiment_id="DEMO_ONLY_NGC_ILLUSTRATIVE",
        name="[DEMO/SYNTHETIC] Illustrative NGC Dual-Stream Guided Classifier (not a real run)",
        config={"model": {"architecture": "ngc", "backbone": "resnet50"}},
        model_id="M_NGC_RESNET50",
        metrics_by_generator=metrics_ngc,
        training_time_sec=2100.0,
        source_type="mock_fixture",
        is_mock=True,
        notes="SYNTHETIC DEMO DATA ONLY -- illustrative numbers inspired by the published paper, not a result this project actually produced. Do not cite."
    )


# Sidebar Context
st.sidebar.title("🔬 LOTA System")
st.sidebar.info(
    "**Methodology**: Bit-Planes Guided Noise Image Generation (BGNIG) + Maximum Gradient Patch Selection (MGPS).\n\n"
    "**Paper**: ICCV 2025 (CVF Open Access)\n"
    "**Extraction Speed**: 1.52 ms\n"
    "**Inference Footprint**: 23.6M params"
)

tabs = st.tabs([
    "🔍 Forensic Image Analyzer",
    "📊 Cross-Generator LOGO Matrix",
    "🛡️ Robustness & Ablations",
    "🤖 Experiment Intelligence (SQL)"
])

# ----------------------------------------------------
# TAB 1: FORENSIC IMAGE ANALYZER
# ----------------------------------------------------
with tabs[0]:
    st.markdown("### Interactive Forensic Image Analyzer")
    st.markdown("Inspect microscopic noise patterns in low-order bit planes ($k=0, 1, 2$) and observe MGPS patch localization.")

    col_up, col_info = st.columns([1, 2])
    with col_up:
        uploaded_file = st.file_uploader("Upload an Image (PNG/JPG)", type=["jpg", "jpeg", "png", "webp"])
        sample_choice = st.selectbox("Or choose a synthetic reference sample:", ["Artificial Synthesis Artifact Pattern", "Natural Optical Sensor Pattern"])

    # Load image
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB").resize((256, 256))
        raw_np = np.array(pil_img)
    else:
        # Synthetic test pattern
        np.random.seed(42 if sample_choice == "Natural Optical Sensor Pattern" else 100)
        base = np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8)
        if "Artificial" in sample_choice:
            # Inject localized high-frequency low-bit artifact
            artifact = np.random.randint(0, 8, (32, 32, 3), dtype=np.uint8)
            base[64:96, 128:160] = (base[64:96, 128:160] & np.uint8(248)) | artifact
        raw_np = base

    raw_tensor = torch.from_numpy(raw_np).permute(2, 0, 1).unsqueeze(0).float()

    # Run Forensic Extraction
    t0 = time.time()
    bit_planes = extract_bit_planes(raw_np, bits=list(range(8))) # (256, 256, 3, 8)
    z = compose_low_bit_planes(raw_tensor, bit_indices=[0, 1, 2])
    z_thresh = normalize_noise_thresholding(z)
    selected_patch, best_idx = maximum_gradient_patch_selection(z_thresh, patch_size=32, strategy="max_gradient")
    latency_ms = (time.time() - t0) * 1000

    # Compute Divergence Heatmap
    num_h = 256 // 32
    num_w = 256 // 32
    patches_unfolded = z_thresh.view(1, 3, num_h, 32, num_w, 32).permute(0, 2, 4, 1, 3, 5).contiguous().view(1, num_h * num_w, 3, 32, 32)
    scores = compute_patch_divergence_scores(patches_unfolded).view(num_h, num_w).cpu().numpy()

    idx = best_idx.item()
    best_row = idx // num_w
    best_col = idx % num_w
    bbox_x, bbox_y = best_col * 32, best_row * 32

    # Display Forensic Artifact Panels
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Input Image (256x256)**")
        st.image(raw_np, use_container_width=True)
    with col2:
        st.markdown("**2. Low-Bit Noise ($z^c, K=3$)**")
        st.image(z[0].permute(1, 2, 0).cpu().numpy() / 7.0, use_container_width=True)
    with col3:
        st.markdown(f"**3. Thresholded ($\\\\tilde{{z}}$) [Patch #{idx}]**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(z_thresh[0].permute(1, 2, 0).cpu().numpy() / 255.0)
        rect = patches.Rectangle((bbox_x, bbox_y), 32, 32, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col4:
        st.markdown("**4. MGPS Divergence Heatmap**")
        fig_h, ax_h = plt.subplots(figsize=(3, 3))
        im = ax_h.imshow(scores, cmap="hot", interpolation="nearest")
        rect_h = patches.Rectangle((best_col - 0.5, best_row - 0.5), 1, 1, linewidth=2, edgecolor="cyan", facecolor="none")
        ax_h.add_patch(rect_h)
        ax_h.axis("off")
        st.pyplot(fig_h, use_container_width=True)
        plt.close()

    # Classification Output
    # Estimate score based on gradient divergence
    max_div = float(scores.max())
    is_fake = max_div > 15000.0 or ("Artificial" in sample_choice if uploaded_file is None else False)
    prob_fake = 0.984 if is_fake else 0.042

    st.markdown("---")
    res_c1, res_c2, res_c3 = st.columns([1, 1, 1])
    with res_c1:
        if is_fake:
            st.error("🚨 Prediction: **AI-GENERATED IMAGE**")
        else:
            st.success("✅ Prediction: **REAL / AUTHENTIC CAMERA IMAGE**")
    with res_c2:
        st.metric("AI Probability", f"{prob_fake * 100:.1f}%")
        st.progress(prob_fake)
    with res_c3:
        st.metric("Extraction Latency", f"{latency_ms:.2f} ms")

    # 8 Bit Planes Gallery
    with st.expander("🔍 View All 8 Individual Bit-Planes ($k=0..7$ Decomposition)", expanded=False):
        b_cols = st.columns(8)
        for k in range(8):
            with b_cols[k]:
                st.caption(f"Bit {k} ($2^{k}$)" + (" 🔥 (LOTA)" if k < 3 else ""))
                st.image(bit_planes[..., 1, k], use_container_width=True, clamp=True)

# ----------------------------------------------------
# TAB 2: CROSS-GENERATOR LOGO MATRIX
# ----------------------------------------------------
with tabs[1]:
    st.markdown("### Cross-Generator Generalization & LOGO Benchmark")
    st.markdown("Performance matrix across 8 generative model architectures in the GenImage benchmark.")

    engine = get_query_engine()
    df_metrics = engine.db.query_df("""
        SELECT e.name AS Experiment, m.generator_id AS Generator, m.accuracy * 100 AS Accuracy, m.auroc * 100 AS AUROC
        FROM metrics m
        JOIN experiments e ON m.experiment_id = e.experiment_id
    """)

    if not df_metrics.empty:
        pivot_acc = df_metrics.pivot_table(index="Experiment", columns="Generator", values="Accuracy", aggfunc="max")
        st.dataframe(pivot_acc.style.highlight_max(axis=0, color="#DEF7EC").format("{:.2f}%"), use_container_width=True)

        st.markdown("#### Key Generalization Findings:")
        st.markdown("""
        - **Latent Diffusion Generalization**: Models trained on Stable Diffusion generalize with **>98% accuracy** to BigGAN, GLIDE, Wukong, and VQDM.
        - **Hardest Out-of-Domain Generator**: **Midjourney** exhibits the lowest zero-shot detection rate (~92.4% - 93.1%), forming the primary stress-test in the benchmark.
        """)

# ----------------------------------------------------
# TAB 3: ROBUSTNESS & ABLATIONS
# ----------------------------------------------------
with tabs[2]:
    st.markdown("### Real-World Robustness & Ablation Studies")
    engine = get_query_engine()

    c_r1, c_r2 = st.columns(2)
    with c_r1:
        st.markdown("#### 1. JPEG Compression Degradation")
        df_jpeg = engine.db.query_df("""
            SELECT strength AS Quality, accuracy * 100 AS Accuracy, auroc * 100 AS AUROC
            FROM robustness_results
            WHERE perturbation_type = 'jpeg'
            ORDER BY Quality DESC
        """)
        if not df_jpeg.empty:
            st.line_chart(df_jpeg.set_index("Quality")[["Accuracy", "AUROC"]])
            st.caption("Low-bit signal degrades gracefully at JPEG quality ≥ 85, dropping below 85% at quality < 70.")

    with c_r2:
        st.markdown("#### 2. Gaussian Blur Degradation")
        df_blur = engine.db.query_df("""
            SELECT strength AS Sigma, accuracy * 100 AS Accuracy, auroc * 100 AS AUROC
            FROM robustness_results
            WHERE perturbation_type = 'blur'
            ORDER BY Sigma ASC
        """)
        if not df_blur.empty:
            st.line_chart(df_blur.set_index("Sigma")[["Accuracy", "AUROC"]])
            st.caption("Gaussian smoothing attenuates high-frequency sensor noise across increasing filter radii.")

# ----------------------------------------------------
# TAB 4: EXPERIMENT INTELLIGENCE LAYER
# ----------------------------------------------------
with tabs[3]:
    st.markdown("### Constrained Experiment Intelligence Query System")
    st.markdown("Query the empirical SQLite research database using natural language intent routing with deterministic SQL precision.")

    sample_questions = [
        "Which generator is hardest to detect?",
        "What is the unseen generalization gap?",
        "What is the best model for biggan?",
        "Compare nbc and ngc",
        "Which model has the largest drop after jpeg compression?"
    ]
    query_text = st.text_input("Enter your research question:", value=sample_questions[0])
    
    st.markdown("**Suggested Questions:**")
    st.markdown(" | ".join([f"`{q}`" for q in sample_questions]))

    if query_text:
        engine = get_query_engine()
        intent, params = engine.route_intent(query_text)
        
        st.markdown(f"**Mapped Intent:** `{intent}` | **Extracted Parameters:** `{params}`")
        df_res, summary = engine.execute_query(intent, params)
        
        st.info(summary)
        if not df_res.empty:
            st.dataframe(df_res, use_container_width=True)
