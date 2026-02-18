
import os
import sys
import time
import cv2
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_utils import BlazeFaceDetector

def benchmark():
    print("="*60)
    print(" SHIELD-RYZEN V2 — BLAZEFACE LATENCY BENCHMARK")
    print("="*60)
    
    try:
        detector = BlazeFaceDetector()
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return
        
    # Dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Warmup
    print("[1] Warmup (10 runs)...")
    for _ in range(10):
        _ = detector.detect_faces(img)
        
    # Benchmark
    print("[2] Benchmarking (100 runs)...")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = detector.detect_faces(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        
    avg = np.mean(times)
    fps = 1000 / avg
    
    print(f"    Avg Latency: {avg:.2f} ms")
    print(f"    FPS:         {fps:.1f}")
    
    if avg < 5.0:
        print("✅ SUCCESS: Latency < 5ms (CPU/NPU verified)")
    elif avg < 15.0:
        print("⚠️  WARNING: Latency < 15ms (Likely CPU)")
    else:
        print("❌ FAIL: Latency too high")

if __name__ == "__main__":
    benchmark()
