"""
Shield-Ryzen V2 — Edge Case Test Suite (TASK 12.6)
==================================================
Integration tests for stability under extreme conditions.
Directly invokes engine logic (synchronous) via _process_frame.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 12 of 14 — Comprehensive Testing
"""

import sys
import os
import time
import json
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Adjust sys.path for modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_engine import ShieldEngine

def test_edge_cases():
    print("Running Edge Case Test Suite...")
    
    config = {
        "camera_id": 0,
        "model_path": "shield_ryzen_int8.onnx",
        "detector_type": "mediapipe",
        # Disable threads for this test (we call _process_frame manually)
        # But ShieldEngine starts camera in __init__? 
        # ShieldEngine.__init__ calls self.camera = ShieldCamera(...)
        # We should patch ShieldCamera to avoid hardware init.
    }
    
    results = {}
    
    # Mock ShieldCamera interaction
    with patch('shield_engine.ShieldCamera') as MockCamClass:
        MockCamClass.return_value.get_health_status.return_value = {"fps": 30, "dropped": 0}
        
        # We also need to mock ShieldFacePipeline if we don't want MediaPipe overhead/errors on noise
        # But testing noise handling is part of the test.
        # So we keep FacePipeline real, unless it crashes.
        
        # Also mock ONNX Runtime if model missing
        with patch('onnxruntime.InferenceSession'):
            
            engine = ShieldEngine(config)
            # Do NOT call engine.start(). We use _process_frame directly.
            
            # Test 1: Corrupted Frame (Random Noise)
            print("Test 1: Corrupted Frame")
            try:
                noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                ts = time.time()
                res = engine._process_frame(noise_frame, ts)
                # Should handle gracefully, likely no faces detected
                if res and res.face_results == []:
                    results["corrupted_frame"] = "PASS"
                else:
                    results["corrupted_frame"] = f"FAIL (Unexpected result: {res})"
            except Exception as e:
                results["corrupted_frame"] = f"CRASH: {e}"

            # Test 2: Dark Frame (All Black)
            print("Test 2: Dark Frame")
            try:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                res = engine._process_frame(black_frame, time.time())
                if res and len(res.face_results) == 0:
                    results["dark_frame"] = "PASS"
                else:
                     results["dark_frame"] = f"FAIL (Found faces in black?)"
            except Exception as e:
                results["dark_frame"] = f"CRASH: {e}"

            # Test 3: Multiple Faces Simulation
            # Need to mock detection to return multiple faces
            print("Test 3: Multiple Faces")
            # We patch the instance's face_pipeline
            original_detect = engine.face_pipeline.detect_faces
            
            # Mock 2 faces
            mock_face1 = MagicMock()
            mock_face1.bbox = (10, 10, 100, 100)
            mock_face1.face_crop_299 = np.zeros((1, 3, 299, 299), dtype=np.float32)
            mock_face1.face_crop_raw = np.zeros((299, 299, 3), dtype=np.uint8)
            mock_face1.landmarks_68 = [MagicMock(x=0,y=0)]*68
            
            mock_face2 = MagicMock()
            mock_face2.bbox = (200, 200, 300, 300)
            mock_face2.face_crop_299 = np.zeros((1, 3, 299, 299), dtype=np.float32)
            mock_face2.face_crop_raw = np.zeros((299, 299, 3), dtype=np.uint8)
            mock_face2.landmarks_68 = [MagicMock(x=0,y=0)]*68
            
            engine.face_pipeline.detect_faces = MagicMock(return_value=[mock_face1, mock_face2])
            
            # We also need to ensure inference doesn't crash on mock crops
            # _process_frame calls _run_inference. 
            # If using ONNX, mock session.run
            # We patched InferenceSession class, so engine.session is a Mock.
            # Configure its run method.
            # Batch size 1? No, we call _run_inference per face.
            engine.session.run.return_value = [np.array([[0.1, 0.9]], dtype=np.float32)] # Real
            
            try:
                res = engine._process_frame(black_frame, time.time())
                count = len(res.face_results) if res else 0
                results["multiple_faces"] = "PASS" if count == 2 else f"FAIL (Got {count})"
            except Exception as e:
                results["multiple_faces"] = f"CRASH: {e}"
                import traceback
                traceback.print_exc()

            # Restore detect
            engine.face_pipeline.detect_faces = original_detect

            # Test 4: Memory Stability (100 frames)
            print("Test 4: Memory Stability")
            import psutil
            process = psutil.Process(os.getpid())
            mem_start = process.memory_info().rss
            
            # Run 50 frames
            try:
                for _ in range(50):
                    engine._process_frame(black_frame, time.time())
                
                mem_end = process.memory_info().rss
                growth_mb = (mem_end - mem_start) / 1024 / 1024
                # Allow < 50MB growth (Python overhead etc)
                results["memory_stability"] = "PASS" if growth_mb < 50 else f"FAIL (Growth {growth_mb:.1f}MB)"
            except Exception as e:
                 results["memory_stability"] = f"CRASH: {e}"

    # Save Results
    results["summary"] = f"Passed {list(results.values()).count('PASS')}/{len(results)}"
    
    report_path = os.path.join("benchmarks", "edge_case_results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"✅ Edge Cases Complete. Saved to {report_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_edge_cases()
