import streamlit as st
import os

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
st.set_page_config(page_title="CaNeSy-eNose", layout="wide")
st.title("🧠⚡ CaNeSy-eNose: Dual-Engine Deployment")

device = torch.device("cpu")

GAS_MAP = {
    0: "Ammonia",
    1: "Acetone",
    2: "Ethanol",
    3: "Methane",
    4: "Carbon Monoxide",
    5: "Clean Air"
}

# =========================================================
# MODEL (UNCHANGED CORE)
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

    # Load model
    model = SensorTransformerXAI()

    model_path = os.path.join(base, "sensor_transformer_xai.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load tensors
    X_test_tensor = torch.load(
        os.path.join(base, "X_test_tensor.pt"),
        map_location=device
    )

    # Load causal object
    with open(os.path.join(base, "causal_effect.pkl"), "rb") as f:
        causal = pickle.load(f)

    return model, X_test_tensor, causal


model, X_test_tensor, CAUSAL_EFFECT = load_assets()

# =========================================================
# PIPELINE
# =========================================================
def run(sample):
    report = {}

    with torch.no_grad():
        logits = model(sample)
        probs = F.softmax(logits, dim=1).numpy().flatten()

    pred = np.argmax(probs)
    conf = np.max(probs)
    uncert = entropy(probs)

    report["pred"] = pred
    report["conf"] = conf
    report["entropy"] = uncert

    # 🔥 CORRECT ATTENTION → SENSOR SIGNATURE
    attn_map = model.get_attention()[0][0].cpu().numpy()  # (128 x 128)

    token_importance = attn_map.mean(axis=0)  # collapse
    token_importance = token_importance.reshape(8, 16)

    report["signature"] = token_importance

    return report


# =========================================================
# UI
# =========================================================
st.sidebar.header("📡 Edge Simulator")

idx = st.sidebar.slider("Select Sample", 0, len(X_test_tensor)-1, 0)
run_btn = st.sidebar.button("Run")

if run_btn:

    sample = X_test_tensor[idx]

    if sample.dim() == 2:
        sample = sample.unsqueeze(0)

    report = run(sample)

    col1, col2 = st.columns([1, 1])

    # ================= LEFT =================
    with col1:
        st.subheader("1️⃣ Vanguard Metrics")

        st.metric("Prediction", GAS_MAP[report["pred"]])
        st.metric("Confidence", f"{report['conf']:.2f}")

        st.metric(
            "Entropy",
            f"{report['entropy']:.2f} ⚠️" if report["entropy"] > 1.3 else f"{report['entropy']:.2f} ✅"
        )

        st.subheader("2️⃣ Final Decision")
        st.success(GAS_MAP[report["pred"]])

    # ================= RIGHT =================
    with col2:

        # -------- Physical --------
        st.subheader("💧 Physical Gas Signature")

        raw = sample.numpy().flatten().reshape(8, 16)

        fig1, ax1 = plt.subplots(figsize=(6, 3))
        im1 = ax1.imshow(raw, cmap="magma")
        ax1.set_title("Sensor Array Activation")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1)

        st.pyplot(fig1)

        # -------- Attention Signature (YOUR CORRECT ONE) --------
        st.subheader("🧠 Neural Attention Signature")

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        im2 = ax2.imshow(report["signature"], cmap="viridis")
        ax2.set_title(f"Attention → {GAS_MAP[report['pred']]}")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2)

        st.pyplot(fig2)

        # -------- Difference --------
        st.subheader("📊 Alignment Error")

        diff = np.abs(report["signature"] - raw)

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        im3 = ax3.imshow(diff, cmap="inferno")
        ax3.set_title("Difference (Attention vs Sensor)")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3)

        st.pyplot(fig3)

        st.metric("Alignment Error", f"{diff.mean():.4f}")

else:
    st.info("Select a sample and click Run.")