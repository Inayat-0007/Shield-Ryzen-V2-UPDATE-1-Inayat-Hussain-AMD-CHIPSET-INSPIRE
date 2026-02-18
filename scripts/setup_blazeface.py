"""
Shield-Ryzen V2 -- BlazeFace NPU Setup
=======================================
Extracts BlazeFace detection model from MediaPipe task file.
Converts TFLite -> ONNX for VitisAI NPU acceleration.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 5 of 12 -- Optimization
"""

import os
import sys
import zipfile
import subprocess
try:
    import tflite2onnx
except ImportError:
    tflite2onnx = None

TASK_FILE = "face_landmarker.task"
TFLITE_PATH = "models/face_detector.tflite"
ONNX_PATH = "models/blazeface.onnx"

def setup_blazeface():
    print("="*60)
    print(" SHIELD-RYZEN V2 — BLAZEFACE NPU SETUP")
    print("="*60)
    
    # 1. Extract TFLite from .task file
    if not os.path.exists(TASK_FILE):
        print(f"❌ Error: {TASK_FILE} not found!")
        sys.exit(1)
        
    print(f"[1] Extracting {TFLITE_PATH} from {TASK_FILE}...")
    with zipfile.ZipFile(TASK_FILE, 'r') as z:
        if "face_detector.tflite" not in z.namelist():
            print("❌ Error: face_detector.tflite not found inside task file!")
            sys.exit(1)
            
        os.makedirs("models", exist_ok=True)
        with z.open("face_detector.tflite") as source, open(TFLITE_PATH, "wb") as target:
            target.write(source.read())
            
    print(f"    Extracted size: {os.path.getsize(TFLITE_PATH)/1024:.2f} KB")
    
    # 2. Convert TFLite -> ONNX
    print(f"[2] Converting to ONNX (for NPU VitisAI)...")
    if tflite2onnx is None:
        print("❌ Error: tflite2onnx not installed!")
        sys.exit(1)
        
    try:
        tflite2onnx.convert(TFLITE_PATH, ONNX_PATH)
        print(f"✅ ONNX model saved to {ONNX_PATH}")
        print(f"   Size: {os.path.getsize(ONNX_PATH)/1024:.2f} KB")
    except Exception as e:
        print(f"⚠️  Conversion failed: {e}")
        print("    Depending on custom ops (TFLite_Detection_PostProcess), this might fail.")
        print("    If failed, we will use a CPU-based fallback in blazeface_detector.py.")
        # We don't exit 1 because we can fallback.

if __name__ == "__main__":
    setup_blazeface()
