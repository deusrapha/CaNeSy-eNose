import onnx2tf
print("Imported onnx2tf")
onnx2tf.convert(
    input_onnx_file_path="mtl_velocity_model.onnx",
    output_folder_path="saved_model",
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True,
)
print("Conversion completed!")
