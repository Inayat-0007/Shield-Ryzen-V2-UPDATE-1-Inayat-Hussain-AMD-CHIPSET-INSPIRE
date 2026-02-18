"""
Shield-Ryzen V2 — Baseline Performance Measurement
====================================================
Captures current system performance BEFORE fixes are applied.
Outputs reproducible JSON for regression testing across all 12 parts.

Usage:
    python benchmarks/baseline.py [--duration 60] [--output benchmarks/baseline_results.json]

Developer: Inayat Hussain | AMD Slingshot 2026
Part 1 of 12 — Measurement Infrastructure
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Add project root to path so we can import project modules
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


# ─── Hardware Info ────────────────────────────────────────────

def _collect_hardware_info() -> dict[str, Any]:
    """Gather reproducible hardware & software environment info."""
    hw: dict[str, Any] = {
        "cpu": platform.processor() or "Unknown",
        "cpu_model": platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2) if psutil else "N/A",
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }

    if ort is not None:
        hw["onnxruntime"] = ort.__version__
        hw["ort_providers"] = ort.get_available_providers()
    else:
        hw["onnxruntime"] = "Not installed"
        hw["ort_providers"] = []

    # Attempt NPU / GPU detection
    try:
        import torch
        hw["torch"] = torch.__version__
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 2
            )
    except ImportError:
        hw["torch"] = "Not installed"
        hw["cuda_available"] = False

    # AMD NPU detection (best-effort)
    try:
        if ort and "VitisAIExecutionProvider" in ort.get_available_providers():
            hw["npu"] = "AMD XDNA (VitisAI detected)"
        else:
            hw["npu"] = "Not detected / CPU fallback"
    except Exception:
        hw["npu"] = "Detection failed"

    return hw


# ─── Camera-Only Benchmark ───────────────────────────────────

def _benchmark_camera_capture(
    duration_seconds: int,
    camera_id: int = 0,
) -> dict[str, Any]:
    """Measure raw camera capture performance (no inference).

    Returns per-frame timing stats and actual FPS.
    """
    from shield_camera import ShieldCamera

    results: dict[str, Any] = {"stage": "camera_capture"}
    frame_times: list[float] = []

    cam = ShieldCamera(camera_id=camera_id)
    if not cam.is_opened():
        results["error"] = "Cannot open camera"
        return results

    start = time.monotonic()
    while (time.monotonic() - start) < duration_seconds:
        t0 = time.perf_counter()
        ok, frame, ts = cam.read_validated_frame()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if ok:
            frame_times.append(elapsed_ms)

    health = cam.get_health_status()
    cam.release()

    if frame_times:
        results["frames_captured"] = len(frame_times)
        results["fps_actual"] = round(health["fps_actual"], 2)
        results["frame_time_ms"] = {
            "mean": round(statistics.mean(frame_times), 3),
            "median": round(statistics.median(frame_times), 3),
            "p95": round(sorted(frame_times)[int(len(frame_times) * 0.95)], 3),
            "p99": round(sorted(frame_times)[int(len(frame_times) * 0.99)], 3),
            "min": round(min(frame_times), 3),
            "max": round(max(frame_times), 3),
            "stdev": round(statistics.stdev(frame_times), 3) if len(frame_times) > 1 else 0.0,
        }
        results["frames_dropped"] = health["frames_dropped"]
        results["drop_rate_pct"] = round(health["drop_rate_pct"], 2)
    else:
        results["error"] = "No valid frames captured"

    return results


# ─── Inference Benchmark ─────────────────────────────────────

def _benchmark_inference(
    n_runs: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    """Benchmark ONNX inference on synthetic input (no camera).

    Measures pure inference latency without I/O overhead.
    """
    results: dict[str, Any] = {"stage": "inference_only"}

    script_dir = Path(__file__).resolve().parent.parent
    int8_path = script_dir / "shield_ryzen_int8.onnx"
    fp32_path = script_dir / "shield_ryzen_v2.onnx"

    if ort is None:
        results["error"] = "onnxruntime not installed"
        return results

    for label, model_path in [("int8", int8_path), ("fp32", fp32_path)]:
        if not model_path.exists():
            results[label] = {"error": f"Model not found: {model_path.name}"}
            continue

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(
                str(model_path), sess_options=sess_opts, providers=providers
            )
            active_provider = session.get_providers()[0]

            dummy = np.random.randn(1, 3, 299, 299).astype(np.float32)

            # Warmup
            for _ in range(warmup):
                session.run(None, {"input": dummy})

            # Timed runs
            times_ms: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                session.run(None, {"input": dummy})
                times_ms.append((time.perf_counter() - t0) * 1000.0)

            results[label] = {
                "provider": active_provider,
                "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
                "runs": n_runs,
                "warmup": warmup,
                "latency_ms": {
                    "mean": round(statistics.mean(times_ms), 3),
                    "median": round(statistics.median(times_ms), 3),
                    "p95": round(sorted(times_ms)[int(len(times_ms) * 0.95)], 3),
                    "p99": round(sorted(times_ms)[int(len(times_ms) * 0.99)], 3),
                    "min": round(min(times_ms), 3),
                    "max": round(max(times_ms), 3),
                    "stdev": round(statistics.stdev(times_ms), 3) if len(times_ms) > 1 else 0.0,
                },
                "theoretical_fps": round(1000.0 / statistics.mean(times_ms), 1),
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


# ─── System Utilization ──────────────────────────────────────

def _sample_utilization(duration_seconds: int = 10) -> dict[str, Any]:
    """Sample CPU and memory utilization over a time window."""
    if psutil is None:
        return {"error": "psutil not installed"}

    cpu_samples: list[float] = []
    mem_samples: list[float] = []

    start = time.monotonic()
    while (time.monotonic() - start) < duration_seconds:
        cpu_samples.append(psutil.cpu_percent(interval=0.5))
        mem_samples.append(psutil.virtual_memory().percent)

    return {
        "duration_seconds": duration_seconds,
        "cpu_percent": {
            "mean": round(statistics.mean(cpu_samples), 1),
            "max": round(max(cpu_samples), 1),
        },
        "memory_percent": {
            "mean": round(statistics.mean(mem_samples), 1),
            "max": round(max(mem_samples), 1),
        },
    }


# ─── Main Entry Point ────────────────────────────────────────

def capture_baseline_metrics(
    duration_seconds: int = 60,
    output_path: str = "benchmarks/baseline_results.json",
) -> dict[str, Any]:
    """Run full baseline measurement suite and save results.

    Measures:
      - Hardware & software environment
      - Camera capture performance
      - ONNX inference latency (INT8 + FP32)
      - System utilization

    Args:
        duration_seconds: How long to run each benchmark stage.
        output_path: Path for the JSON results file.

    Returns:
        Complete results dictionary.
    """
    print("=" * 60)
    print("  SHIELD-RYZEN V2 — BASELINE PERFORMANCE CAPTURE")
    print("=" * 60)

    results: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration_seconds,
            "version": "V2-Part1",
        },
    }

    # 1. Hardware info
    print("\n[1/4] Collecting hardware info...")
    results["hardware"] = _collect_hardware_info()
    print(f"       CPU: {results['hardware']['cpu']}")
    print(f"       RAM: {results['hardware']['ram_gb']} GB")

    # 2. Camera benchmark
    print(f"\n[2/4] Benchmarking camera capture ({duration_seconds}s)...")
    results["camera"] = _benchmark_camera_capture(duration_seconds)
    if "fps_actual" in results["camera"]:
        print(f"       Actual FPS: {results['camera']['fps_actual']}")
        print(f"       Frame time: {results['camera']['frame_time_ms']['mean']:.1f} ms (mean)")
        print(f"       Drops: {results['camera']['frames_dropped']}")
    else:
        print(f"       ⚠️  {results['camera'].get('error', 'Unknown error')}")

    # 3. Inference benchmark
    print("\n[3/4] Benchmarking ONNX inference (200 runs each)...")
    results["inference"] = _benchmark_inference()
    for label in ["int8", "fp32"]:
        if label in results["inference"] and "latency_ms" in results["inference"][label]:
            info = results["inference"][label]
            print(
                f"       {label.upper()}: {info['latency_ms']['mean']:.2f} ms "
                f"({info['theoretical_fps']} FPS) via {info['provider']}"
            )

    # 4. System utilization
    print("\n[4/4] Sampling system utilization (10s)...")
    results["utilization"] = _sample_utilization(10)
    if "cpu_percent" in results["utilization"]:
        print(f"       CPU: {results['utilization']['cpu_percent']['mean']}% avg")
        print(f"       MEM: {results['utilization']['memory_percent']['mean']}% avg")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Baseline saved → {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shield-Ryzen Baseline Benchmark")
    parser.add_argument("--duration", type=int, default=60, help="Seconds per benchmark stage")
    parser.add_argument("--output", type=str, default="benchmarks/baseline_results.json")
    args = parser.parse_args()

    capture_baseline_metrics(
        duration_seconds=args.duration,
        output_path=args.output,
    )
