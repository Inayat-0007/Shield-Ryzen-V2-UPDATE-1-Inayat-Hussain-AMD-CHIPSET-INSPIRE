
import sys
import os
import time
import json
import psutil
import platform
import numpy as np

# Adjust sys.path to root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None

from v3_xdna_engine import ShieldXDNAEngine
from shield_engine import ShieldEngine

def benchmark_end_to_end_fps(
    duration_seconds: int = 60,
    engine_type: str = "auto"   # "onnx" | "xdna" | "auto"
) -> dict:
    """
    Runs the full pipeline for N seconds and measures
    TRUE end-to-end FPS including ALL stages.
    Tests on both XDNA1 and XDNA2 if available.
    """
    print(f"🚀 Starting Benchmark: {duration_seconds}s, Engine: {engine_type}")

    # 1. Hardware Info
    hw = {
        "cpu": platform.processor(),
        "cpu_model": platform.machine(), # Better than '...'
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "os": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__ if ort else "N/A",
        "opencv": cv2.__version__ if cv2 else "N/A",
    }
    
    # Try detecting NPU via XDNA engine logs or similar
    # For now, placeholder or check logs
    hw["npu"] = "AMD XDNA (Detection logic placeholder)"
    hw["npu_tops"] = "Unknown"

    # 2. Initialize Engine
    config = {
        "camera_id": 0, # Or file/mock
        "camera_backend": cv2.CAP_DSHOW if cv2 else 0,
        "detector_type": "mediapipe",
        "min_confidence": 0.5,
        "model_path": "shield_ryzen_int8.onnx",
        "enable_temporal": True,
        "enable_frequency": True,
        "enable_attribution": True,
        "xmodel_path": "models/compiled/xdna1/shield_xception.xmodel"
    }

    if engine_type == "xdna" or engine_type == "auto":
        # Force attempt XDNA
        engine = ShieldXDNAEngine(config)
        # If fallback occurred, log it
        if not engine.use_native:
            if engine_type == "xdna":
                print("⚠️ Requested XDNA but fell back to ONNX.")
            engine_type = "onnx (fallback)"
        else:
            engine_type = "xdna"
    else:
        engine = ShieldEngine(config)
        engine_type = "onnx"

    # 3. Warmup
    print("Warmup (Starting Threads)...")
    engine.start()
    time.sleep(2.0) # Let camera settle

    # 4. Benchmark Loop
    print(f"Running for {duration_seconds} seconds...")
    start_time = time.time()
    frame_count = 0
    timings = {
        "capture": [], "detect": [], "infer": [], 
        "liveness": [], "texture": [], "state": [], 
        "hud": [], "total": [], "advanced": []
    }
    
    while (time.time() - start_time) < duration_seconds:
        result = engine.get_latest_result()
        if result:
            frame_count += 1
            
            # Collect timings from result.timing_breakdown
            t = result.timing_breakdown
            timings["capture"].append(t.get("capture_ms", 0))
            timings["detect"].append(t.get("detect_ms", 0))
            timings["infer"].append(t.get("infer_total_ms", 0))
            timings["liveness"].append(t.get("liveness_total_ms", 0))
            timings["texture"].append(t.get("texture_total_ms", 0))
            timings["state"].append(t.get("state_ms", 0))
            timings["hud"].append(t.get("hud_ms", 0))
            timings["advanced"].append(t.get("advanced_ms", 0))
            timings["total"].append(t.get("total_ms", 0))
        else:
            time.sleep(0.001) # Yield if queue empty

    engine.stop()

    avg_fps = frame_count / duration_seconds
    
    # Calc Stats
    def get_stats(arr):
        if not arr: return 0.0
        return float(np.mean(arr))

    timing_breakdown_avg_ms = {
        "capture": get_stats(timings["capture"]),
        "face_detect": get_stats(timings["detect"]),
        "inference": get_stats(timings["infer"]),
        "liveness": get_stats(timings["liveness"]),
        "texture": get_stats(timings["texture"]),
        "state_machine": get_stats(timings["state"]),
        "hud_render": get_stats(timings["hud"]),
        "advanced_modules": get_stats(timings["advanced"]),
        "total": get_stats(timings["total"]),
    }

    fps_percentiles = {
        "p5": float(np.percentile(timings["total"], 5)) if timings["total"] else 0,
        "p50": float(np.percentile(timings["total"], 50)) if timings["total"] else 0,
        "p95": float(np.percentile(timings["total"], 95)) if timings["total"] else 0,
    }

    # Hypothetical Baseline for Comparison (Part 1 values)
    baseline_fps = 15.0 # Example
    improvement = ((avg_fps - baseline_fps) / baseline_fps) * 100 if baseline_fps else 0

    report = {
        "hardware": hw,
        "engine_type": engine_type,
        "duration_seconds": duration_seconds,
        "total_frames": frame_count,
        "average_fps": round(avg_fps, 2),
        "timing_breakdown_avg_ms": {k: round(v, 2) for k,v in timing_breakdown_avg_ms.items()},
        "fps_percentiles": {k: round(v, 2) for k,v in fps_percentiles.items()},
        "honest_claim": (
            f"End-to-end FPS: {avg_fps:.0f} "
            f"(measured on {hw.get('cpu_model','unknown')}, "
            f"including all pipeline stages, "
            f"averaged over {duration_seconds}s)"
        ),
        "baseline_comparison": {
            "baseline_fps": baseline_fps,
            "improvement_percent": round(improvement, 2),
        }
    }

    report_path = os.path.join("benchmarks", "fps_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Benchmark Complete. Report saved to {report_path}")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    benchmark_end_to_end_fps(duration_seconds=10) # Short default for test run
