import time
import json
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, send_from_directory

app = Flask(__name__, static_folder='static')

# Load the ONNX Model once
ONNX_PATH = "mtl_velocity_model.onnx"
print(f"Loading {ONNX_PATH}...")
try:
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

CLASSES = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']

def generate_sensor_stream():
    """Simulates real-time sensor polling and runs inference at a defined rate."""
    # Simulation state
    t = 0
    buffer = np.zeros((1, 50, 16), dtype=np.float32)
    velocity = 2.5 # initial wind velocity m/s
    
    # We will loop infinitely to simulate streaming
    while True:
        # Simulate new sensor reading (16 channels)
        # We'll make it somewhat smooth by using sine waves + noise, and inject an "event" periodically
        is_event = (t % 200) > 100 # event every 200 ticks, lasting 100 ticks
        
        base_signal = np.sin(t * 0.1) * 0.5 + 0.5
        noise = np.random.randn(16) * 0.1
        
        if is_event:
            # Sudden spike in readings to simulate gas exposure
            new_reading = (base_signal + 5.0 + noise).astype(np.float32)
            velocity = max(0.5, velocity + np.random.randn() * 0.1)
        else:
            new_reading = (noise).astype(np.float32)
            velocity = 2.5 + np.sin(t * 0.05) * 0.5
            
        # Shift buffer left and append new reading at the end
        buffer[0, :-1, :] = buffer[0, 1:, :]
        buffer[0, -1, :] = new_reading
        
        # Run ONNX Inference if model is loaded
        predicted_state = "Unknown"
        co_ppm = 0.0
        eth_ppm = 0.0
        latency_ms = 0.0
        
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
            
            class_idx = np.argmax(class_logits[0])
            predicted_state = CLASSES[class_idx]
            
            co_ppm = float(regression_ppm[0][0])
            eth_ppm = float(regression_ppm[0][1])
        
        # Build JSON payload
        payload = {
            "timestamp": t,
            "latency_ms": round(latency_ms, 2),
            "predicted_state": predicted_state,
            "co_ppm": max(0, round(co_ppm, 2)),
            "eth_ppm": max(0, round(eth_ppm, 2)),
            "velocity": round(velocity, 2),
            "sensor_mean": round(float(np.mean(new_reading)), 3)
        }
        
        yield f"data: {json.dumps(payload)}\n\n"
        
        t += 1
        # Sleep to simulate ~10Hz update rate (100ms) for UI smoothness
        # While real sensors might be 100Hz, we can send updates at 10Hz to the frontend
        time.sleep(0.1)

@app.route('/stream')
def stream():
    return Response(generate_sensor_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    print("Starting PC Simulation Server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)
