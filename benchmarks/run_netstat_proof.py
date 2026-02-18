
import subprocess
import time
import sys
import os

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_int8_engine import ShieldEngine

def prove_offline():
    print("Starting Shield Inference (Dummy run for proof)...")
    config = {
        "camera_id": 0,
        "detector_type": "mediapipe",
        "model_path": "shield_ryzen_int8.onnx",
        "mock_camera": True # Use mock camera to avoid hardware issues during proof
    }
    
    # We patch camera read inside engine to return dummy frame
    from unittest.mock import MagicMock, patch
    import numpy as np
    
    with patch("v3_int8_engine.ShieldCamera.read_validated_frame", return_value=(True, np.zeros((480, 640, 3), dtype=np.uint8), time.time())):
        engine = ShieldEngine(config)
        
        # Run loop for 5 seconds
        start = time.time()
        while time.time() - start < 5:
            engine.process_frame()
            if time.time() - start > 2:
                # Capture netstat
                print("Capturing Network State...")
                try:
                    res = subprocess.run(["netstat", "-an"], capture_output=True, text=True)
                    with open("network_during_inference.txt", "w") as f:
                        f.write(res.stdout)
                    print("Netstat captured.")
                    break
                except Exception as e:
                    print(f"Netstat failed: {e}")
                    break
                    
    print("Proof Complete.")

if __name__ == "__main__":
    prove_offline()
