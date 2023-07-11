# Define the path to save the ONNX file
onnx_file_path = 'path_to_save_model.onnx'

# Export the model to ONNX format
torch.onnx.export(model, example_input, onnx_file_path, opset_version=11)