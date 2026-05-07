import torch
import os
from velocity_model_training import MTL_TemporalTransformer_Velocity

def export_velocity_model_to_onnx():
    print("Initializing Velocity-Aware Architecture...")
    model = MTL_TemporalTransformer_Velocity(num_sensors=16, d_model=16, window_size=50)
    
    # Normally we would load the trained weights here:
    # model.load_state_dict(torch.load('best_velocity_model.pth'))

    # Load trained weights before export
    if os.path.exists("best_velocity_model.pth"):
        model.load_state_dict(torch.load("best_velocity_model.pth", map_location="cpu"))
        print("Loaded trained weights from best_velocity_model.pth")
    else:
        print("Warning: best_velocity_model.pth not found. Exporting untrained model.")

    model.eval()
    

    # Dummy inputs for ONNX tracing
    # 1. The sensor window (Batch=1, Time=100, Sensors=16)
    dummy_x = torch.randn(1, 50, 16)
    
    # 2. The physical velocity input from the robot's speedometer (Batch=1, Dim=1)
    # E.g., 2.5 meters per second
    dummy_v = torch.tensor([[2.5]], dtype=torch.float32)

    onnx_path = "mtl_velocity_model.onnx"
    print(f"Exporting model to {onnx_path}...")
    
    # Export the model
    torch.onnx.export(
        model, 
        (dummy_x, dummy_v), 
        onnx_path,
        export_params=True,
        opset_version=18,  # Required for some Transformer operations
        do_constant_folding=True,
        input_names=['sensor_window', 'velocity'],
        output_names=['class_logits', 'regression_ppm']
    )
    print("✅ Successfully exported Velocity-Aware ONNX model for Raspberry Pi!")
    
    # Optional: Display ONNX model size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX Model Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_velocity_model_to_onnx()
