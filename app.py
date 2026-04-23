import os
import time
import pickle

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

# =========================================================
# ENVIRONMENT FIX
# =========================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CaNeSy-eNose Dual-Engine Deployment",
    layout="wide"
)

st.title("🧠 CaNeSy-eNose: Split-Edge Neural Deployment")
st.markdown(
    """
    This application simulates a **Dual-Engine Intelligent Electronic Nose Pipeline**.

    ### Pipeline
    - **Vanguard Model** → Fast edge inference
    - **Analyst Model** → Attention extraction + interpretability
    - **Active Learning Oracle** → Human validation for uncertain or unknown gases

    The system escalates to the Human Oracle when:
    - Entropy exceeds threshold
    - Confidence is too low
    - Symbolic sanity constraints fail
    - Unknown gas distributions appear
    """
)

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cpu")

# =========================================================
# GAS LABEL MAP
# =========================================================
GAS_MAP = {
    0: "Ethanol",
    1: "Ethylene",
    2: "Ammonia",
    3: "Acetaldehyde",
    4: "Acetone",
    5: "Toluene"
}

# =========================================================
# MODEL
# =========================================================
class SensorTransformerXAI(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, num_classes=6):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 128, d_model))

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])

        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)
        self.attention_weights = []

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding

        self.attention_weights = []

        for attn, ff in zip(self.attn_layers, self.ff_layers):
            attn_out, attn_w = attn(x, x, x)
            x = x + attn_out
            x = x + ff(x)
            self.attention_weights.append(attn_w.detach().cpu())

        x = x.mean(dim=1)
        return self.classifier(x)

    def get_attention(self):
        return self.attention_weights

# =========================================================
# HELPERS
# =========================================================
def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))


def safe_scalar_causal_effect(obj):
    try:
        if isinstance(obj, (int, float, np.floating)):
            return float(obj)

        if isinstance(obj, np.ndarray) and obj.size == 1:
            return float(obj.item())

        if torch.is_tensor(obj) and obj.numel() == 1:
            return float(obj.item())
    except Exception:
        pass

    return None


def sample_to_2d(sample_tensor):
    arr = sample_tensor.detach().cpu().numpy().flatten()

    if arr.size != 128:
        return None

    return arr.reshape(8, 16)


def collapse_attention_to_signature(attn_map):
    if attn_map is None:
        return np.zeros((8, 16))

    token_importance = attn_map.mean(axis=0)

    if token_importance.size != 128:
        return np.zeros((8, 16))

    return token_importance.reshape(8, 16)


def get_top_feature_evidence(sample_tensor, attn_map, top_k=10):
    raw_flat = sample_tensor.detach().cpu().numpy().flatten()

    if raw_flat.size != 128:
        return []

    if attn_map is None:
        attn_importance = np.zeros(128)
    else:
        attn_importance = attn_map.mean(axis=0)

        if attn_importance.size != 128:
            attn_importance = np.zeros(128)

    raw_abs = np.abs(raw_flat)
    raw_norm = raw_abs / (raw_abs.max() + 1e-8)

    attn_abs = np.abs(attn_importance)
    attn_norm = attn_abs / (attn_abs.max() + 1e-8)

    combined = 0.5 * raw_norm + 0.5 * attn_norm

    top_idx = np.argsort(combined)[::-1][:top_k]

    evidence = []

    for idx in top_idx:
        row = idx // 16
        col = idx % 16

        evidence.append({
            "Feature": f"F{idx+1} (R{row+1}, C{col+1})",
            "Index": int(idx),
            "Sensor Row": int(row + 1),
            "Sensor Column": int(col + 1),
            "Raw Value": float(raw_flat[idx]),
            "Attention Importance": float(attn_importance[idx]),
            "Combined Score": float(combined[idx])
        })

    return evidence


def format_probability_table(probs):
    rows = []

    for i, p in enumerate(probs):
        rows.append({
            "Class": GAS_MAP[i],
            "Probability": float(p)
        })

    rows = sorted(rows, key=lambda x: x["Probability"], reverse=True)

    return rows

# =========================================================
# LOAD ASSETS
# =========================================================
@st.cache_resource
def load_assets():
    base = get_base_dir()

    model_path = os.path.join(base, "sensor_transformer_xai.pth")
    tensor_path = os.path.join(base, "X_test_tensor.pt")
    causal_path = os.path.join(base, "causal_effect.pkl")

    analyst_model = SensorTransformerXAI()
    analyst_model.load_state_dict(torch.load(model_path, map_location=device))
    analyst_model.to(device)
    analyst_model.eval()

    vanguard_model = SensorTransformerXAI()
    vanguard_model.load_state_dict(torch.load(model_path, map_location=device))
    vanguard_model.to(device)
    vanguard_model.eval()

    X_test_tensor = torch.load(tensor_path, map_location=device)

    with open(causal_path, "rb") as f:
        causal_effect = pickle.load(f)

    return vanguard_model, analyst_model, X_test_tensor, causal_effect


vanguard, analyst, X_test_tensor, CAUSAL_EFFECT = load_assets()

# =========================================================
# SESSION STATE
# =========================================================
if "oracle_trigger" not in st.session_state:
    st.session_state.oracle_trigger = False

if "oracle_resolution" not in st.session_state:
    st.session_state.oracle_resolution = None

if "current_sample_idx" not in st.session_state:
    st.session_state.current_sample_idx = 0

# =========================================================
# PIPELINE
# =========================================================
def run_dual_engine(sample_tensor, entropy_thresh=1.3, confidence_thresh=0.40):

    report = {
        "timings": {},
        "flags": [],
        "needs_oracle": False
    }

    # ---------------- Vanguard ----------------
    t0 = time.perf_counter()

    with torch.no_grad():
        vanguard_logits = vanguard(sample_tensor)
        vanguard_probs = F.softmax(vanguard_logits, dim=1).detach().cpu().numpy().flatten()

    report["timings"]["Edge Vanguard Inference"] = time.perf_counter() - t0

    pred_class = int(np.argmax(vanguard_probs))
    confidence = float(np.max(vanguard_probs))
    uncert = float(entropy(vanguard_probs))

    report["vanguard_pred"] = pred_class
    report["vanguard_conf"] = confidence
    report["entropy"] = uncert
    report["vanguard_probs"] = vanguard_probs

    needs_analyst = False

    # ---------------- OOD Detection ----------------
    if uncert > entropy_thresh or confidence < confidence_thresh:
        report["flags"].append("⚠️ HIGH OOD ENTROPY: Unknown gas distribution pattern.")
        needs_analyst = True

    # ---------------- Symbolic Rule ----------------
    if pred_class == 2 and torch.sum(torch.abs(sample_tensor)).item() < 0.1:
        report["flags"].append("🛑 SYMBOLIC OVERRIDE: Impossible physical pattern for Ammonia.")
        needs_analyst = True

    # ---------------- Analyst ----------------
    t0 = time.perf_counter()

    with torch.no_grad():
        analyst_logits = analyst(sample_tensor)
        analyst_probs = F.softmax(analyst_logits, dim=1).detach().cpu().numpy().flatten()

    attn_list = analyst.get_attention()

    attn_map = None

    if len(attn_list) > 0:
        try:
            attn_map = attn_list[0][0].cpu().numpy()
        except Exception:
            attn_map = None

    report["timings"]["Analyst XAI Extraction"] = time.perf_counter() - t0

    report["analyst_pred"] = int(np.argmax(analyst_probs))
    report["analyst_conf"] = float(np.max(analyst_probs))

    report["attention"] = attn_map
    report["signature"] = collapse_attention_to_signature(attn_map)

    report["top_feature_evidence"] = get_top_feature_evidence(
        sample_tensor.squeeze(0),
        attn_map,
        top_k=10
    )

    report["probability_table"] = format_probability_table(vanguard_probs)

    # ---------------- Causal Signal ----------------
    causal_scalar = safe_scalar_causal_effect(CAUSAL_EFFECT)

    if causal_scalar is not None:
        if abs(causal_scalar) < 0.05:
            report["flags"].append("Causal linkage weak for top sensor features.")

    report["needs_oracle"] = needs_analyst

    return report

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("📡 Edge Device Simulator")

sample_idx = st.sidebar.slider(
    "Select Test Sample",
    min_value=0,
    max_value=len(X_test_tensor) - 1,
    value=st.session_state.current_sample_idx
)

if sample_idx != st.session_state.current_sample_idx:
    st.session_state.current_sample_idx = sample_idx
    st.session_state.oracle_trigger = False
    st.session_state.oracle_resolution = None

trigger_anomaly = st.sidebar.checkbox(
    "Inject Unknown Gas Noise",
    help="Creates an unseen gas pattern."
)

entropy_threshold = st.sidebar.slider(
    "OOD Entropy Threshold",
    min_value=0.5,
    max_value=2.5,
    value=1.3,
    step=0.05
)

run_btn = st.sidebar.button("Run Pipeline")

# =========================================================
# MAIN EXECUTION
# =========================================================
if run_btn or st.session_state.oracle_trigger:

    sample = X_test_tensor[sample_idx].clone()

    if trigger_anomaly:
        sample += torch.randn_like(sample) * 5.0

    if sample.dim() == 2:
        sample = sample.unsqueeze(0)

    report = run_dual_engine(sample, entropy_thresh=entropy_threshold)

    # =====================================================
    # ORACLE INTERVENTION
    # =====================================================
    if report["needs_oracle"] and st.session_state.oracle_resolution is None:

        st.session_state.oracle_trigger = True

        st.error("🚨 Vanguard detected anomaly or low confidence.")

        if report["flags"]:
            st.warning(" | ".join(report["flags"]))

        st.write("### 🧑‍🔬 Active Learning: Human Oracle Required")
        st.write(
            "Review the sensor evidence, strongest features, and probability distribution before assigning a label."
        )

        top_class = GAS_MAP[report["vanguard_pred"]]

        st.markdown(
            f"""
            ### Oracle Briefing

            - Suggested Class: **{top_class}**
            - Confidence: **{report['vanguard_conf']:.2f}**
            - Entropy: **{report['entropy']:.2f}**
            - Analyst Trigger Reason: **{' | '.join(report['flags']) if report['flags'] else 'Uncertain prediction'}**
            """
        )

        raw_grid = sample_to_2d(sample.squeeze(0))

        if raw_grid is not None:
            fig, ax = plt.subplots(figsize=(7, 3))
            im = ax.imshow(raw_grid, cmap="magma")
            ax.set_title("Physical Gas Signature")
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        st.write("### 🔍 Top Features Driving the Decision")

        evidence_df = pd.DataFrame(report["top_feature_evidence"])
        st.dataframe(evidence_df, use_container_width=True)

        st.write("### 📈 Class Probability Distribution")

        prob_df = pd.DataFrame(report["probability_table"])
        st.dataframe(prob_df, use_container_width=True)

        user_correction = st.selectbox(
            "Assign Final Human Label",
            ["Unknown/Anomaly"] + list(GAS_MAP.values())
        )

        if st.button("Confirm Oracle Override & Resume"):
            st.session_state.oracle_resolution = user_correction
            st.session_state.oracle_trigger = False
            st.rerun()

        st.stop()

    # =====================================================
    # DASHBOARD
    # =====================================================
    col1, col2 = st.columns([1, 1])

    raw_grid = sample_to_2d(sample.squeeze(0))
    signature_grid = report["signature"]

    # =====================================================
    # LEFT PANEL
    # =====================================================
    with col1:

        st.subheader("1️⃣ Vanguard Metrics")

        st.metric("Prediction", GAS_MAP[report["vanguard_pred"]])
        st.metric("Confidence", f"{report['vanguard_conf']:.2f}")

        st.metric(
            "Entropy",
            f"{report['entropy']:.2f} ⚠️" if report["entropy"] > entropy_threshold else f"{report['entropy']:.2f} ✅"
        )

        if report["flags"]:
            st.subheader("Flags")
            for flag in report["flags"]:
                st.warning(flag)

        st.subheader("2️⃣ Final Decision")

        if st.session_state.oracle_resolution:
            final_gas = st.session_state.oracle_resolution
            st.info("Decision overridden by Human Oracle")
        else:
            final_gas = GAS_MAP[report["vanguard_pred"]]

        st.success(f"Final Verdict: {final_gas}")

        if final_gas == "Unknown/Anomaly":
            st.warning("⚠️ NEW UNKNOWN GAS DETECTED")
            st.info(
                "This sample should be physically verified and later added to the training pipeline."
            )

        st.subheader("Top Feature Evidence")

        evidence_df = pd.DataFrame(report["top_feature_evidence"])
        st.dataframe(evidence_df, use_container_width=True)

    # =====================================================
    # RIGHT PANEL
    # =====================================================
    with col2:

        tab1, tab2, tab3 = st.tabs([
            "💧 Physical Signature",
            "🧠 Attention Signature",
            "📊 Alignment Error"
        ])

        with tab1:
            if raw_grid is not None:
                fig1, ax1 = plt.subplots(figsize=(6, 3))
                im1 = ax1.imshow(raw_grid, cmap="magma")
                ax1.set_title("Sensor Activation")
                ax1.axis("off")
                plt.colorbar(im1, ax=ax1)
                st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            im2 = ax2.imshow(signature_grid, cmap="viridis")
            ax2.set_title("Analyst Attention Signature")
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2)
            st.pyplot(fig2)

        with tab3:
            if raw_grid is not None:
                diff = np.abs(signature_grid - raw_grid)

                fig3, ax3 = plt.subplots(figsize=(6, 3))
                im3 = ax3.imshow(diff, cmap="inferno")
                ax3.set_title("Attention vs Sensor Difference")
                ax3.axis("off")
                plt.colorbar(im3, ax=ax3)
                st.pyplot(fig3)

                st.metric("Alignment Error", f"{diff.mean():.4f}")

    # =====================================================
    # LATENCY
    # =====================================================
    st.divider()

    st.subheader("Hardware Simulation Latency")

    cols = st.columns(len(report["timings"]))

    for col, (stage, t) in zip(cols, report["timings"].items()):
        with col:
            st.metric(stage, f"{t*1000:.3f} ms")

else:
    st.info("Select a sample and click Run Pipeline.")