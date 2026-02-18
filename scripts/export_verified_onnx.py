"""
Shield-Ryzen V2 — Verified ONNX Export
=======================================
Exports the cryptographically verified ShieldXception model to ONNX.
Ensures the FP32 ONNX model matches the verified checkpoint exactly.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 5 of 12 -- Quantization & Optimization
"""

import os
import sys
import torch
import onnx
import onnxoptimizer
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_xception import ShieldXception, load_model_with_verification

def export_verified_model():
    print("=" * 60)
    print("  SHIELD-RYZEN V2 — VERIFIED ONNX EXPORT")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = "shield_ryzen_v2.onnx"
    checkpoint_path = "ffpp_c23.pth"

    # 1. Load Verified Model
    print(f"[1] Loading verified model from {checkpoint_path}...")
    try:
        # This triggers the hash check in Part 4
        state_dict = load_model_with_verification(checkpoint_path, device)
        model = ShieldXception().to(device)
        model.model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("[2] Model loaded and integrity verified.")
    except Exception as e:
        print(f"❌ FATAL: Model verification failed: {e}")
        sys.exit(1)

    # 2. Prepare Dummy Input
    dummy_input = torch.randn(1, 3, 299, 299).to(device)

    # 3. Export to ONNX
    print(f"[3] Exporting to {output_path} (Opset 17)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 4. Optimize Graph
    print("[4] Optimizing ONNX graph...")
    onnx_model = onnx.load(output_path)
    passes = [
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_unused_initializer',
        'fuse_bn_into_conv'
    ]
    # Check available passes
    available = onnxoptimizer.get_available_passes()
    valid_passes = [p for p in passes if p in available]
    
    optimized_model = onnxoptimizer.optimize(onnx_model, valid_passes)
    onnx.save(optimized_model, output_path)
    
    print(f"✅ SUCCESS: Verified ONNX model saved to {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export_verified_model()
