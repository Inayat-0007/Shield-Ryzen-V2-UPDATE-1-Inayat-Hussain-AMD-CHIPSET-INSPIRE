import onnx
import sys

MODEL = "models/blazeface.onnx"

try:
    model = onnx.load(MODEL)
    print(f"Model Inputs:")
    for i in model.graph.input:
        print(f"  {i.name}: {i.type.tensor_type.shape}")

    print(f"Model Outputs:")
    for o in model.graph.output:
        print(f"  {o.name}: {o.type.tensor_type.shape}")
        
except Exception as e:
    print(f"Error loading {MODEL}: {e}")
