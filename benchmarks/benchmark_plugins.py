"""
Shield-Ryzen V2 — Plugin Performance Benchmark (TASK 12.5)
==========================================================
Measures the computational cost of each security plugin.
Reports contribution to total latency.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 12 of 14 — Comprehensive Testing
"""

import sys
import os
import time
import json
import numpy as np

# Add root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_engine import ShieldEngine

def benchmark_plugins(duration_per_plugin=10):
    print("🚀 Starting Plugin Benchmark...")
    
    # List of plugins to test
    # Key: Config name to enable
    scenarios = [
        ("Baseline (No Plugins)", {}),
        ("Heartbeat (rPPG)", {"enable_heartbeat": True}),
        ("Frequency Analysis (HFER)", {"enable_frequency_analysis": True}),
        ("Codec Forensics (Blocking)", {"enable_codec_forensics": True}),
        ("Adversarial Patch", {"enable_adversarial_detection": True}),
        ("Full Security Suite", {
            "enable_heartbeat": True,
            "enable_frequency_analysis": True,
            "enable_codec_forensics": True,
            "enable_adversarial_detection": True
        })
    ]
    
    results = {}
    
    # Base Config (ALL OFF)
    base_config = {
        "camera_id": 0, # Should ideally use video file for consistency
        "detector_type": "mediapipe",
        "model_path": "shield_ryzen_int8.onnx",
        "enable_challenge_response": False,
        "enable_heartbeat": False,
        "enable_stereo_depth": False,
        "enable_skin_reflectance": False,
        "enable_frequency_analysis": False,
        "enable_codec_forensics": False,
        "enable_adversarial_detection": False,
        "enable_lip_sync": False
    }
    
    for name, overrides in scenarios:
        print(f"\nTesting: {name}")
        cfg = base_config.copy()
        cfg.update(overrides)
        
        try:
            engine = ShieldEngine(cfg)
            engine.start()
            time.sleep(2.0) # Warmup
            
            latencies = []
            
            start_t = time.time()
            frame_count = 0
            
            while (time.time() - start_t) < duration_per_plugin:
                res = engine.get_latest_result()
                if res:
                    frame_count += 1
                    # Total MS for this frame
                    # timing_breakdown might miss 'total_ms' if not computed?
                    # Face processing time.
                    # We look at 'advanced_ms' or 'total_ms'
                    # Or measure queue fetch rate (FPS)
                    
                    # We rely on 'total_ms' reported by engine
                    latencies.append(res.timing_breakdown.get("total_ms", 0))
                else:
                    time.sleep(0.001)
            
            engine.stop()
            
            # Stats
            avg_lat = np.mean(latencies) if latencies else 0
            p95_lat = np.percentile(latencies, 95) if latencies else 0
            fps = frame_count / duration_per_plugin
            
            print(f"  FPS: {fps:.1f}")
            print(f"  Latency (avg): {avg_lat:.2f}ms")
            
            results[name] = {
                "fps": round(fps, 1),
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95_lat, 2)
            }
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {"error": str(e)}

    # Calc Cost
    baseline = results.get("Baseline (No Plugins)", {})
    base_lat = baseline.get("avg_latency_ms", 0)
    
    for name, data in results.items():
        if "avg_latency_ms" in data and base_lat > 0:
            cost = data["avg_latency_ms"] - base_lat
            data["cost_ms"] = round(max(0, cost), 2)
            
    # Save
    report_path = os.path.join("benchmarks", "plugin_benchmark.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Plugin Benchmark Complete. Saved to {report_path}")

if __name__ == "__main__":
    benchmark_plugins()
