import time
import os
import torch
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from velocity_model_training import MTL_TemporalTransformer_Velocity

def main():
    print("=== Phase 1: Model Comparison (PyTorch vs ONNX vs TFLite) ===")

    # 1. Setup Data
    print("\n[1] Generating dummy input data...")
    batch_size = 1
    window_size = 50
    num_sensors = 16
  
    
    # PyTorch inputs
    dummy_x_pt = torch.randn(batch_size, window_size, num_sensors)
    dummy_v_pt = torch.tensor([[2.5]], dtype=torch.float32)
    
    # Numpy inputs for ORT and TFLite
    dummy_x_np = dummy_x_pt.numpy()
    dummy_v_np = dummy_v_pt.numpy()

    # 2. PyTorch Model
    print("\n[2] Loading PyTorch Model...")
    pt_model = MTL_TemporalTransformer_Velocity(num_sensors=num_sensors, d_model=16, window_size=window_size)
    # Attempt to load weights if they exist
    if os.path.exists("best_velocity_model.pth"):
        pt_model.load_state_dict(torch.load("best_velocity_model.pth", map_location='cpu'))
    pt_model.eval()

    print("Running PyTorch inference...")
    start_time = time.time()
    with torch.no_grad():
        pt_out_class, pt_out_reg = pt_model(dummy_x_pt, dummy_v_pt)
    pt_time = time.time() - start_time
    print(f"PyTorch Time: {pt_time*1000:.2f} ms")

    # 3. ONNX Model
    onnx_path = "mtl_velocity_model.onnx"
    print(f"\n[3] Loading ONNX Model from {onnx_path}...")
    if not os.path.exists(onnx_path):
        print("ONNX model not found. Please run export_onnx.py first.")
        return

    ort_session = ort.InferenceSession(onnx_path)
    
    print("Running ONNX inference...")
    start_time = time.time()
    ort_inputs = {
        'sensor_window': dummy_x_np,
        'velocity': dummy_v_np
    }
    ort_outs = ort_session.run(None, ort_inputs)
    ort_time = time.time() - start_time
    print(f"ONNX Time: {ort_time*1000:.2f} ms")

    # 4. Convert ONNX to TFLite
    print("\n[4] Converting ONNX to TFLite using onnx2tf...")
    # onnx2tf typically creates a saved_model directory
    if not os.path.exists("saved_model/model_float32.tflite") and not os.path.exists("saved_model/mtl_velocity_model_float32.tflite"):
        print("Running onnx2tf conversion... (this may take a minute)")
        os.system(f"onnx2tf -i {onnx_path}")
        
    tflite_model_file = "saved_model/mtl_velocity_model_float32.tflite"
    if not os.path.exists(tflite_model_file):
        if os.path.exists("saved_model/model_float32.tflite"):
            tflite_model_file = "saved_model/model_float32.tflite"
        else:
            print("TFLite model not found after conversion. Please check onnx2tf output.")
            return

    # 5. TFLite Model
    print(f"\n[5] Loading TFLite Model from {tflite_model_file}...")
    interpreter = Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_x_tflite = np.transpose(dummy_x_np, (0, 2, 1)).astype(np.float32)

    for detail in input_details:
        if 'sensor_window' in detail['name'] or list(detail['shape']) == [1, 16, 50]:
            interpreter.set_tensor(detail['index'], dummy_x_tflite)
        elif 'velocity' in detail['name'] or list(detail['shape']) == [1, 1]:
            interpreter.set_tensor(detail['index'], dummy_v_np)

    print("Running TFLite inference...")
    start_time = time.time()
    interpreter.invoke()
    tflite_time = time.time() - start_time
    print(f"TFLite Time: {tflite_time*1000:.2f} ms")

    tflite_out_0 = interpreter.get_tensor(output_details[0]['index'])
    tflite_out_1 = interpreter.get_tensor(output_details[1]['index'])
    
    if tflite_out_0.shape[-1] == 1:
        tflite_out_reg = tflite_out_0
        tflite_out_class = tflite_out_1
    else:
        tflite_out_class = tflite_out_0
        tflite_out_reg = tflite_out_1

    # In case ONNX outputs are flipped
    if ort_outs[0].shape[-1] == 1:
        ort_out_reg = ort_outs[0]
        ort_out_class = ort_outs[1]
    else:
        ort_out_class = ort_outs[0]
        ort_out_reg = ort_outs[1]

    print("\n=== Results Comparison ===")
    print(f"PyTorch Outputs: \n  Class: {pt_out_class.numpy()[0]}, \n  Reg: {pt_out_reg.numpy()[0]}")
    print(f"ONNX Outputs: \n  Class: {ort_out_class[0]}, \n  Reg: {ort_out_reg[0]}")
    print(f"TFLite Outputs: \n  Class: {tflite_out_class[0]}, \n  Reg: {tflite_out_reg[0]}")
    
    print("\nMax absolute differences:")
    print(f"PyTorch vs ONNX (Class): {np.max(np.abs(pt_out_class.numpy() - ort_out_class)):.6f}")
    print(f"PyTorch vs ONNX (Reg): {np.max(np.abs(pt_out_reg.numpy() - ort_out_reg)):.6f}")
    print(f"PyTorch vs TFLite (Class): {np.max(np.abs(pt_out_class.numpy() - tflite_out_class)):.6f}")
    print(f"PyTorch vs TFLite (Reg): {np.max(np.abs(pt_out_reg.numpy() - tflite_out_reg)):.6f}")

    print("\n=== Performance Comparison ===")
    print(f"PyTorch: {pt_time*1000:.2f} ms")
    print(f"ONNX:    {ort_time*1000:.2f} ms")
    print(f"TFLite:  {tflite_time*1000:.2f} ms")

if __name__ == "__main__":
    main()
