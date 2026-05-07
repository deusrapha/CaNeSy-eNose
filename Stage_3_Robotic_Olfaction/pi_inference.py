import onnxruntime as ort
import numpy as np
import time

def run_pi_inference():
    onnx_path = "mtl_velocity_model.onnx"
    print(f"Loading {onnx_path} on Raspberry Pi CPU...")
    
    # Initialize ONNX Runtime session
    # On a Raspberry Pi, execution providers would just be ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # In a real scenario, this data comes from physical sensors via I2C/SPI
    print("Reading 100-step sensor window and physical anemometer...")
    dummy_sensor_data = np.random.randn(1, 100, 16).astype(np.float32)
    dummy_velocity = np.array([[2.5]], dtype=np.float32) # 2.5 m/s

    # Run inference
    print("Running Inference...")
    start_time = time.time()
    
    outputs = session.run(
        output_names=['class_logits', 'regression_ppm'],
        input_feed={
            'sensor_window': dummy_sensor_data,
            'velocity': dummy_velocity
        }
    )
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    class_logits, regression_ppm = outputs
    
    # Post-process
    class_idx = np.argmax(class_logits[0])
    classes = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']
    predicted_state = classes[class_idx]
    
    co_ppm = regression_ppm[0][0]
    eth_ppm = regression_ppm[0][1]
    
    print("\n=== ROBOTIC OLFACTION INFERENCE RESULT ===")
    print(f"Latency:        {latency_ms:.2f} ms")
    print(f"Detected State: {predicted_state}")
    print(f"Estimated CO:   {co_ppm:.2f} ppm")
    print(f"Estimated Eth:  {eth_ppm:.2f} ppm")
    print("==========================================\n")
    
    # Gradient Ascent Logic
    print("Sending motor commands based on Gradient Ascent...")
    if eth_ppm > 0.5:
        print("-> Ethylene detected! Moving forward to climb gradient.")
    elif class_idx == 0:
        print("-> Clean Air. Initiating spiraling cast to locate plume.")

if __name__ == "__main__":
    run_pi_inference()
