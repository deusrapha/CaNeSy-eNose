import streamlit as st
import os

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from scipy.stats import entropy

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="CaNeSy-eNose Dual-Engine Deployment", layout="wide")
st.title("🧠 CaNeSy-eNose: Split-Edge Neural Deployment")
st.markdown(
    "Simulation of the **Dual-Engine Architecture**. "
    "The fast **Vanguard Edge Model** runs primary inference. "
    "If uncertainty spikes or a symbolic rule is violated, the heavier "
    "**Analyst Model** engages for deeper inspection and active-learning support."
)

device = torch.device("cpu")

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
# LOAD
# =========================================================
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)

    analyst_model = SensorTransformerXAI()
    analyst_model.load_state_dict(
        torch.load(os.path.join(base, "sensor_transformer_xai.pth"), map_location=device)
    )
    analyst_model.to(device)
    analyst_model.eval()

    vanguard_model = SensorTransformerXAI()
    vanguard_model.load_state_dict(
        torch.load(os.path.join(base, "sensor_transformer_xai.pth"), map_location=device)
    )
    vanguard_model.to(device)
    vanguard_model.eval()

    X_test_tensor = torch.load(
        os.path.join(base, "X_test_tensor.pt"),
        map_location=device
    )

    with open(os.path.join(base, "causal_effect.pkl"), "rb") as f:
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
# DUAL-ENGINE PIPELINE RUNNER
# =========================================================
def run_dual_engine(sample_tensor, entropy_thresh=1.5):
    report = {"timings": {}, "flags": []}

    # Vanguard inference
    t0 = time.perf_counter()
    with torch.no_grad():
        vanguard_logits = vanguard(sample_tensor)
        vanguard_probs = F.softmax(vanguard_logits, dim=1).cpu().numpy().flatten()
    report["timings"]["Edge Vanguard Inference"] = time.perf_counter() - t0

    uncert = entropy(vanguard_probs)
    pred_class = int(np.argmax(vanguard_probs))
    conf = float(np.max(vanguard_probs))

    report["vanguard_pred"] = pred_class
    report["vanguard_conf"] = conf
    report["entropy"] = uncert

    # OOD / symbolic escalation logic
    needs_analyst = False
    if uncert > entropy_thresh or conf < 0.40:
        report["flags"].append("⚠️ HIGH OOD ENTROPY: Unknown gas distribution pattern.")
        needs_analyst = True

    # symbolic override example
    if pred_class == 2 and torch.sum(torch.abs(sample_tensor)).item() < 0.1:
        report["flags"].append("🛑 SYMBOLIC OVERRIDE: Impossible physical signal for Ammonia.")
        needs_analyst = True

    report["needs_oracle"] = needs_analyst

    # Analyst pass
    t0 = time.perf_counter()
    with torch.no_grad():
        analyst_logits = analyst(sample_tensor)
        analyst_probs = F.softmax(analyst_logits, dim=1).cpu().numpy().flatten()

        attn = analyst.get_attention()
        if len(attn) > 0:
            raw_attn = attn[0][0].cpu().numpy()
        else:
            raw_attn = np.zeros((128, 128))

    report["timings"]["Analyst XAI Extraction"] = time.perf_counter() - t0
    report["analyst_pred"] = int(np.argmax(analyst_probs))
    report["analyst_conf"] = float(np.max(analyst_probs))
    report["attention"] = raw_attn

    # 8x16 collapsed attention signature for display
    token_importance = raw_attn.mean(axis=0)
    report["signature"] = token_importance.reshape(8, 16)

    # causal warning
    try:
        causal_value = float(CAUSAL_EFFECT)
        if abs(causal_value) < 0.05:
            report["flags"].append("Causal linkage weak for top sensor features.")
    except Exception:
        report["flags"].append("Causal effect object loaded, but could not be reduced to a scalar warning check.")

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
    "Inject Simulated Unknown Gas Noise",
    help="Perturbs the sensor pattern to simulate an unseen gas."
)

run_btn = st.sidebar.button("Run Deployment Pipeline")

# =========================================================
# MAIN APP
# =========================================================
if run_btn or st.session_state.oracle_trigger:
    sample = X_test_tensor[sample_idx].clone()

    if trigger_anomaly:
        sample += torch.randn_like(sample) * 5.0

    if sample.dim() == 2:
        sample = sample.unsqueeze(0)

    report = run_dual_engine(sample, entropy_thresh=1.3)

    # Active learning / oracle pause
    if report["needs_oracle"] and st.session_state.oracle_resolution is None:
        st.session_state.oracle_trigger = True

        st.error("🚨 Vanguard detected anomaly or low confidence. Pipeline paused.")
        if report["flags"]:
            st.warning(" | ".join(report["flags"]))

        st.write("### 🧑‍🔬 Active Learning: Human Oracle Required")
        st.write("Review the physical signature and assign a label.")

        fig, ax = plt.subplots(figsize=(6, 3))
        signature_array = sample.squeeze(0).cpu().numpy().flatten()

        if len(signature_array) == 128:
            im = ax.imshow(signature_array.reshape(8, 16), cmap="magma")
            ax.set_title("Physical Gas Signature")
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        user_correction = st.selectbox(
            "Assign True Gas Class:",
            ["Unknown/Anomaly"] + list(GAS_MAP.values())
        )

        if st.button("Confirm Oracle Override & Resume"):
            st.session_state.oracle_resolution = user_correction
            st.session_state.oracle_trigger = False
            st.rerun()

        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1️⃣ Vanguard Metrics")
        st.metric("Raw Edge Prediction", GAS_MAP[report["vanguard_pred"]])
        st.metric("Model Confidence", f"{report['vanguard_conf']:.2f}")
        st.metric(
            "OOD Entropy Score",
            f"{report['entropy']:.2f} ⚠️" if report["entropy"] > 1.3 else f"{report['entropy']:.2f} ✅"
        )

        if report["flags"]:
            st.subheader("Flags")
            for flag in report["flags"]:
                st.warning(flag)

        st.subheader("2️⃣ Final Decision")

        if st.session_state.oracle_resolution:
            final_gas_str = st.session_state.oracle_resolution
            st.info("ACTIVE LEARNING: Overridden by Human Oracle.")
            st.success(f"Final Verdict: {final_gas_str}")
        else:
            final_gas_str = GAS_MAP[report["vanguard_pred"]]
            st.success("EDGE CONFIDENT: Handled entirely by fast edge node.")
            st.success(f"Final Verdict: {final_gas_str}")

        if final_gas_str == "Unknown/Anomaly":
            st.warning("⚠️ NEW UNKNOWN GAS DETECTED")
            st.info(
                "This signature does not match any known profile. "
                "Submit the physical sample to a domain expert for external validation "
                "and future active-learning updates."
            )

    with col2:
        tab1, tab2, tab3 = st.tabs([
            "💧 Physical Gas Signature",
            "🧠 Neural Attention Signature",
            "📊 Alignment Error"
        ])

        raw = sample.squeeze(0).cpu().numpy().flatten().reshape(8, 16)

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            im1 = ax1.imshow(raw, cmap="magma")
            ax1.set_title("Sensor Array Activation")
            ax1.axis("off")
            plt.colorbar(im1, ax=ax1)
            st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            im2 = ax2.imshow(report["signature"], cmap="viridis")
            ax2.set_title("Collapsed Analyst Attention Signature")
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2)
            st.pyplot(fig2)

        with tab3:
            diff = np.abs(report["signature"] - raw)
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            im3 = ax3.imshow(diff, cmap="inferno")
            ax3.set_title("Difference (Attention vs Sensor)")
            ax3.axis("off")
            plt.colorbar(im3, ax=ax3)
            st.pyplot(fig3)

            st.metric("Alignment Error", f"{diff.mean():.4f}")

    st.divider()
    st.subheader("Hardware Simulation Latency")

    latency_cols = st.columns(len(report["timings"]))
    for col, (stage, t) in zip(latency_cols, report["timings"].items()):
        with col:
            st.metric(stage, f"{t * 1000:.3f} ms")

else:
    st.info("Select a test sample and click Run.")