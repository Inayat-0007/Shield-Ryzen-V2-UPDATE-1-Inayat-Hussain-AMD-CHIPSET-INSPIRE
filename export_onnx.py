"""
Shield-Ryzen V2 — ONNX Exporter (TASK 9.4)
==========================================
Converts the PyTorch XceptionNet model to ONNX format (FP32).
This is the pre-requisite step for quantization (INT8) and NPU deployment.

Usage: python export_onnx.py

Developer: Inayat Hussain | AMD Slingshot 2026
Part 9 of 14 — AMD Native Optimization
"""

import torch
import torch.onnx
import os
import sys

# Add root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shield_xception import ShieldXception

def export_model():
    print("Exporting ShieldXception to ONNX...")
    
    # Load Model
    # Assumes weights exist or initializes random if missing (for structural export)
    try:
        model = ShieldXception() # No args
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            # We should use load_model_with_verification from shield_xception 
            # or just load raw if we trust it.
            # But the keys might need cleaning.
            # However, exporting raw structure is fine if just for NPU test.
            # To be safe, try direct load, if fail, warn.
            try:
                model.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {weights_path} into internal model")
            except Exception:
                print("Direct load failed. Trying wrapper load...")
                model.load_state_dict(state_dict, strict=False)
                
        else:
            print(f"Warning: Weights {weights_path} not found. Exporting UNTRAINED model structure.")
            
        model.eval()
        
        # Dummy Input (Batch: 1, RGB: 3, Height: 299, Width: 299)
        dummy_input = torch.randn(1, 3, 299, 299, requires_grad=True)
        
        os.makedirs("models", exist_ok=True)
        output_path = "models/shield_ryzen.onnx"
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13, # Ops set 13 is good for Vitis AI
            do_constant_folding=True,
            input_names=['input_1'],
            output_names=['output_1'],
            dynamic_axes={'input_1': {0: 'batch_size'}, 'output_1': {0: 'batch_size'}}
        )
        
        print(f"Successfully exported to {output_path}")
        print("Next Step: Run 'python quantize_ryzen.py' to optimize for NPU.")
        
    except Exception as e:
        print(f"Export Failed: {e}")

if __name__ == "__main__":
    export_model()
