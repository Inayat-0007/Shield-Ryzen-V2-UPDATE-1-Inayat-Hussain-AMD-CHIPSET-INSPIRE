
import sys
import os
import json
import subprocess
import time

# Metrics
from benchmark_fps import benchmark_end_to_end_fps
from benchmark_accuracy import test_accuracy
from benchmark_threshold import run_threshold_analysis
from benchmark_power import measure_power
from edge_case_test_suite import test_edge_cases

def run_final_report():
    print("═══════════════════════════════════════════════════════════════════")
    print("SHIELD-RYZEN V2 — FINAL PERFORMANCE VALIDATION")
    print("═══════════════════════════════════════════════════════════════════")

    # 1. End-to-End FPS
    print("\n[1/7] Running FPS Benchmark...")
    # Run twice: Once ONNX, Once XDNA (if avl)
    fps_onnx = benchmark_end_to_end_fps(duration_seconds=10, engine_type="onnx")
    fps_xdna = benchmark_end_to_end_fps(duration_seconds=10, engine_type="xdna")

    # 2. Accuracy
    print("\n[2/7] Running Accuracy Benchmark...")
    test_accuracy({
        "FF++ c23": "datasets/ffpp_c23", 
        "Celeb-DF": "datasets/celeb_df"
    }, mock=True)

    # 3. Thresholds
    print("\n[3/7] Analyzing Thresholds...")
    run_threshold_analysis(mock=True)

    # 4. Power
    print("\n[4/7] Measuring Power...")
    measure_power(duration_seconds=5)

    # 5. Edge Cases
    print("\n[5/7] Testing Edge Cases...")
    test_edge_cases()

    # 6. Unit Tests (Pytest)
    print("\n[6/7] Running Unit Tests...")
    try:
        # Run from root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # -v for verbose, --tb=short for concise
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=root_dir,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        with open("test_results.txt", "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        print(f"Tests finished with code {result.returncode}")
    except Exception as e:
        print(f"Failed to run pytest: {e}")

    # 7. Dependencies
    print("\n[7/7] Freezing Requirements...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True
        )
        with open("requirements_locked.txt", "w") as f:
            f.write(result.stdout)
        print("Requirements saved.")
    except Exception as e:
        print(f"Failed to freeze pip: {e}")

    print("\n═══════════════════════════════════════════════════════════════════")
    print("VALIDATION COMPLETE. ARTIFACTS GENERATED.")
    print("═══════════════════════════════════════════════════════════════════")

if __name__ == "__main__":
    run_final_report()
