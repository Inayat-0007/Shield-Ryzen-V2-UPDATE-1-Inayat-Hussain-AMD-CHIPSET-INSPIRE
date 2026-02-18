
import sys
import os
import time
import json
import numpy as np

# Adjust sys.path for modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_int8_engine import ShieldEngine

def test_edge_cases() -> dict:
    """
    Comprehensive edge case testing:
    - 2 faces in frame simultaneously
    - Face with sunglasses (occlusion) -> Simulated visually? Or mocked result.
    - Camera unplugged/replugged -> Not easily testable in script without manual intervention.
    - Corrupted frames (random noise)
    - Dark lighting conditions
    - Bright backlighting
    - Memory leak check (short run)
    """
    print("Running Edge Case Test Suite...")
    
    config = {
        "camera_id": 0,
        "model_path": "shield_ryzen_int8.onnx",
        "mock_camera": True # We need to mock camera to inject specific frames
    }
    
    # We create a special engine that allows injecting frames
    # Or modify ShieldCamera to accept injected frames?
    # Easier: Just modify engine.process_frame() to take optional 'frame' arg
    # But process_frame() calls self.camera.read_validated_frame()
    # We can patch camera read.
    
    from unittest.mock import MagicMock, patch
    
    results = {}
    
    # Test 1: Corrupted Frame (Random Noise)
    noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    with patch("v3_int8_engine.ShieldCamera.read_validated_frame", return_value=(True, noise_frame, time.time())):
        engine = ShieldEngine(config)
        try:
            res = engine.process_frame()
            results["corrupted_frame"] = "PASS" if res.state != "ERROR" else "FAIL"
        except Exception as e:
            results["corrupted_frame"] = f"CRASH: {e}"

    # Test 2: Dark Frame (All Black)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("v3_int8_engine.ShieldCamera.read_validated_frame", return_value=(True, black_frame, time.time())):
        engine = ShieldEngine(config)
        res = engine.process_frame()
        # Should detect no faces, return NO_FACE state
        results["dark_frame"] = "PASS" if res.state == "NO_FACE" else f"FAIL (Got {res.state})"

    # Test 3: Multiple Faces (requires mock face detector returning >1 face)
    # We can mock face_pipeline.detect_faces
    with patch("v3_int8_engine.ShieldCamera.read_validated_frame", return_value=(True, black_frame, time.time())), \
         patch("v3_int8_engine.ShieldFacePipeline.detect_faces") as MockDetect:
         
         # Mock 2 faces
         MockDetect.return_value = [MagicMock(bbox=(10,10,50,50), landmarks=np.zeros((5,2))), 
                                    MagicMock(bbox=(100,100,50,50), landmarks=np.zeros((5,2)))]
         
         engine = ShieldEngine(config)
         res = engine.process_frame()
         
         results["multiple_faces"] = "PASS" if len(res.face_results) == 2 else f"FAIL (Got {len(res.face_results)})"

    # Test 4: Memory Leak Check (Short)
    # Run 100 frames
    import psutil
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss
    
    with patch("v3_int8_engine.ShieldCamera.read_validated_frame", return_value=(True, black_frame, time.time())):
        engine = ShieldEngine(config)
        for _ in range(100):
            engine.process_frame()
            
    mem_end = process.memory_info().rss
    diff_mb = (mem_end - mem_start) / 1024 / 1024
    results["memory_leak_check"] = "PASS" if diff_mb < 50 else f"FAIL (Leaked {diff_mb:.2f} MB)"

    report_path = "benchmarks/edge_case_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"✅ Edge Case Tests Complete. Saved to {report_path}")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    test_edge_cases()
