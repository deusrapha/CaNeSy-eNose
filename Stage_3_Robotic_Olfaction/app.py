import streamlit as st
import time
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.stats import entropy

# Streamlit App Configuration
st.set_page_config(page_title="CaNeSy-eNose Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize ONNX Model
@st.cache_resource
def load_model():
    ONNX_PATH = "mtl_velocity_model.onnx"
    try:
        return ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        st.error(f"Error loading ONNX model. Please ensure '{ONNX_PATH}' is in the repository. Details: {e}")
        return None

session = load_model()
CLASSES = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']

st.title("CaNeSy-eNose System")
st.markdown("**🟢 Connected to Edge (ONNX - Simulated Stream)**")

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

if st.button("Start/Stop Simulation"):
    st.session_state.run_simulation = not st.session_state.run_simulation

# UI Placeholders
alert_placeholder = st.empty()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Real-Time Concentration Regression")
    chart_placeholder = st.empty()

with col2:
    class_placeholder = st.empty()
    st.markdown("---")
    velocity_placeholder = st.empty()
    st.markdown("---")
    latency_placeholder = st.empty()
    st.markdown("---")
    uncertainty_placeholder = st.empty()
    st.markdown("---")
    action_placeholder = st.empty()

# Simulation state
if "t" not in st.session_state:
    st.session_state.t = 0
    st.session_state.buffer = np.zeros((1, 50, 16), dtype=np.float32)
    st.session_state.velocity = 2.5
    st.session_state.history_co = []
    st.session_state.history_eth = []

max_data_points = 60

if st.session_state.run_simulation:
    t = st.session_state.t
    buffer = st.session_state.buffer
    velocity = st.session_state.velocity
    
    is_event = (t % 200) > 100
    base_signal = np.sin(t * 0.1) * 0.5 + 0.5
    noise = np.random.randn(16) * 0.1
    
    if is_event:
        new_reading = (base_signal + 5.0 + noise).astype(np.float32)
        velocity = max(0.5, velocity + np.random.randn() * 0.1)
    else:
        new_reading = (noise).astype(np.float32)
        velocity = 2.5 + np.sin(t * 0.05) * 0.5
        
    buffer[0, :-1, :] = buffer[0, 1:, :]
    buffer[0, -1, :] = new_reading
    
    predicted_state = "Unknown"
    co_ppm = 0.0
    eth_ppm = 0.0
    latency_ms = 0.0
    uncertainty_score = 0.0
    agent_action = "Standby"
    
    if session is not None:
        start_time = time.time()
        v_input = np.array([[velocity]], dtype=np.float32)
        outputs = session.run(
            output_names=['class_logits', 'regression_ppm'],
            input_feed={
                'sensor_window': buffer,
                'velocity': v_input
            }
        )
        latency_ms = (time.time() - start_time) * 1000
        class_logits, regression_ppm = outputs
        
        exp_logits = np.exp(class_logits[0] - np.max(class_logits[0]))
        probs = exp_logits / exp_logits.sum()
        class_idx = np.argmax(probs)
        predicted_state = CLASSES[class_idx]
        
        uncertainty_score = float(entropy(probs))
        if uncertainty_score < 0.5:
            agent_action = "Accept"
        elif uncertainty_score < 1.0:
            agent_action = "Re-sample"
        else:
            agent_action = "Escalate (Human-in-the-loop)"
            
        co_ppm = max(0.0, float(regression_ppm[0][0]))
        eth_ppm = max(0.0, float(regression_ppm[0][1]))

    # Update History
    st.session_state.history_co.append(co_ppm)
    st.session_state.history_eth.append(eth_ppm)
    
    if len(st.session_state.history_co) > max_data_points:
        st.session_state.history_co.pop(0)
        st.session_state.history_eth.pop(0)

    # Update UI
    if co_ppm > 5.0 or eth_ppm > 5.0:
        alert_placeholder.error("⚠️ TOXIC GAS CONCENTRATION DETECTED! INITIATING GRADIENT ASCENT OVERRIDE.")
    else:
        alert_placeholder.empty()

    df = pd.DataFrame({
        "CO (ppm)": st.session_state.history_co,
        "Ethylene (ppm)": st.session_state.history_eth
    })
    chart_placeholder.line_chart(df, color=["#f87171", "#38bdf8"])
    
    class_placeholder.metric("Gas Classification", predicted_state)
    velocity_placeholder.metric("Environmental Overlay (Wind)", f"{velocity:.2f} m/s")
    latency_placeholder.metric("Inference Latency", f"{latency_ms:.1f} ms")
    uncertainty_placeholder.metric("Uncertainty Score (Entropy)", f"{uncertainty_score:.3f}")
    action_placeholder.metric("Agentic Controller Action", agent_action)

    st.session_state.t += 1
    st.session_state.velocity = velocity
    time.sleep(0.1)
    st.rerun()
else:
    # Render static state when not running
    df = pd.DataFrame({
        "CO (ppm)": st.session_state.history_co if st.session_state.history_co else [0],
        "Ethylene (ppm)": st.session_state.history_eth if st.session_state.history_eth else [0]
    })
    chart_placeholder.line_chart(df, color=["#f87171", "#38bdf8"])
    class_placeholder.metric("Gas Classification", "Standby")
    velocity_placeholder.metric("Environmental Overlay (Wind)", f"{st.session_state.velocity:.2f} m/s")
    latency_placeholder.metric("Inference Latency", "0.0 ms")
    uncertainty_placeholder.metric("Uncertainty Score (Entropy)", "0.000")
    action_placeholder.metric("Agentic Controller Action", "Standby")
