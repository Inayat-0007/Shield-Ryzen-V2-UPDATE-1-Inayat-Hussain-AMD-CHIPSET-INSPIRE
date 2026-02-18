"""
Shield-Ryzen V2 -- Quantization Tests
======================================
Verifies model quantization quality, file sizes, and NPU compatibility.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 5 of 12 -- Optimization
"""

import os
import json
import glob
import pytest
import numpy as np

# Paths
CALIBRATION_DIR = "data/calibration_set_v2"
MODEL_FP32 = "shield_ryzen_v2.onnx"
MODEL_INT8 = "shield_ryzen_int8.onnx"
MODEL_INT4 = "models/shield_xception_int4.onnx"
REPORT_PATH = "quantization_report.json"
LOG_PATH = "logs/npu_execution.log"

def test_calibration_dataset_has_minimum_500_frames():
    """Verify sufficient calibration data exists (Part 1 requirement)."""
    images = glob.glob(os.path.join(CALIBRATION_DIR, "*.jpg")) + \
             glob.glob(os.path.join(CALIBRATION_DIR, "*.png"))
    # Allow slightly less if some failed to load? 521 found.
    assert len(images) >= 500, f"Found only {len(images)} calibration images"

def test_int8_model_smaller_than_fp32():
    """Verify quantization reduced file size."""
    assert os.path.exists(MODEL_FP32), "FP32 model missing"
    assert os.path.exists(MODEL_INT8), "INT8 model missing"
    
    fp32_size = os.path.getsize(MODEL_FP32)
    int8_size = os.path.getsize(MODEL_INT8)
    
    assert int8_size < fp32_size * 0.5, "INT8 model should be < 50% of FP32 size"

def test_compression_ratio_within_expected_range():
    """Verify compression matches theoretical expectations (approx 75%)."""
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
        
    compression = report["compression_percent"]
    # Theoretical max 75%, usually 70-75% due to overhead
    assert 60.0 <= compression <= 80.0, f"Unexpected compression: {compression:.1f}%"

def test_prediction_agreement_above_99_percent():
    """Verify quantization didn't break accuracy on test subset."""
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
        
    agreement = report["accuracy_comparison"]["agreement_rate"]
    assert agreement >= 99.0, f"Accuracy agreement too low: {agreement}%"

def test_per_channel_quantization_used():
    """Verify configuration used per-channel quantization (better for CNNs)."""
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
        
    assert report["per_channel_quantization"] is True

def test_npu_execution_coverage_logged():
    """Verify NPU compatibility analysis was logged."""
    assert os.path.exists(LOG_PATH), "NPU execution log missing"
    
    with open(LOG_PATH, "r") as f:
        log = json.load(f)
        
    # Check for QDQ nodes (indicates NPU readiness even if running on CPU)
    assert "qdq_nodes" in log
    assert log["qdq_nodes"] > 0, "No QDQ nodes found! Model not quantized for NPU."

def test_quantization_report_generated():
    """Verify comprehensive report generation."""
    assert os.path.exists(REPORT_PATH)
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
    
    expected_keys = [
        "fp32_size_mb", "int8_size_mb", "compression_percent",
        "compression_honest_description", "quantization_format",
        "calibration_frames", "per_channel_quantization",
        "accuracy_comparison", "npu_verification"
    ]
    for k in expected_keys:
        assert k in report, f"Missing report key: {k}"

def test_int4_model_exists():
    """Verify INT4 model generation (Task 5.2)."""
    assert os.path.exists(MODEL_INT4), "INT4 model missing"
    # Size check vs FP32
    int4_size = os.path.getsize(MODEL_INT4)
    fp32_size = os.path.getsize(MODEL_FP32)
    assert int4_size < fp32_size * 0.5, "INT4 model not compressed properly"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
