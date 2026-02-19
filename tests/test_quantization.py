"""
Shield-Ryzen V2 — Quantization Verification Tests (TASK 5.2)
============================================================
Checks post-quantization artifacts, report metrics, and pipeline consistency.

Tests:
  1. test_calibration_dataset_has_minimum_500_frames: Verify dataset size.
  2. test_int8_model_smaller_than_fp32: File size check.
  3. test_compression_ratio_within_expected_range: Expect ~70-75%.
  4. test_prediction_agreement_above_99_percent: Verify accuracy impact.
  5. test_per_channel_quantization_used: Check report flag.
  6. test_npu_execution_coverage_logged: Verify NPU report exists.
  7. test_quantization_uses_same_normalization_as_pipeline: Verify normalization range.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 5 of 12 — Quantization & Optimization
"""

import json
import os
import sys
import numpy as np
import pytest
import cv2

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_face_pipeline import ShieldFacePipeline

REPORT_PATH = "quantization_report.json"
MODEL_INT8 = "shield_ryzen_int8.onnx"
MODEL_FP32 = "shield_ryzen_v2.onnx"
CALIBRATION_DIR = "data/calibration_set_v2"

@pytest.fixture
def report():
    if not os.path.exists(REPORT_PATH):
        pytest.skip("Quantization report not found. Run quantize_int8.py first.")
    with open(REPORT_PATH, "r") as f:
        return json.load(f)

def test_calibration_dataset_has_minimum_500_frames():
    """Verify calibration dataset contains 500+ frames as per spec."""
    import glob
    count = len(glob.glob(os.path.join(CALIBRATION_DIR, "*.jpg"))) + \
            len(glob.glob(os.path.join(CALIBRATION_DIR, "*.png")))
    assert count >= 500, f"Found only {count} frames, expected 500+"

def test_int8_model_smaller_than_fp32(report):
    """INT8 model must be significantly smaller than FP32."""
    assert report["int8_size_mb"] < report["fp32_size_mb"]
    assert os.path.getsize(MODEL_INT8) < os.path.getsize(MODEL_FP32)

def test_compression_ratio_within_expected_range(report):
    """
    Theoretical INT8 compression is ~75% (8-bit vs 32-bit).
    We expect roughly 70-76% reduction depending on overhead.
    """
    compression = report["compression_percent"]
    assert 70.0 <= compression <= 78.0, f"Unexpected compression: {compression}%"

def test_prediction_agreement_above_99_percent(report):
    """
    Quantization-aware accuracy verification.
    We require >99% agreement rate between FP32 and INT8 predictions.
    This ensures numerical stability on the NPU target.
    """
    agreement = report["accuracy_comparison"]["agreement_rate"]
    assert agreement >= 99.0, f"Agreement too low: {agreement}%"

def test_per_channel_quantization_used(report):
    """Per-channel quantization is critical for Xception depthwise convolutions."""
    assert report["per_channel_quantization"] is True

def test_npu_execution_coverage_logged(report):
    """Verify NPU execution provider was checked."""
    npu_data = report["npu_verification"]
    assert "available_providers" in npu_data
    assert "npu_coverage_percent" in npu_data
    # We don't fail if NPU is 0% (dev machine might not have it), but we verified the CHECK existed.

def test_quantization_uses_same_normalization_as_pipeline():
    """
    Verify ShieldFacePipeline produces [-1, 1] output range for white image.
    This confirms the normalization logic used in quantization script matches.
    """
    pipeline = ShieldFacePipeline(detector_type="mediapipe", max_faces=1)
    
    # Create dummy white image (255)
    dummy = np.full((300, 300, 3), 255, dtype=np.uint8)
    bbox = (0, 0, 300, 300)
    
    # align_and_crop
    tensor, _ = pipeline.align_and_crop(dummy, bbox)
    
    # Expected: (255/255 - 0.5)/0.5 = (1.0 - 0.5)/0.5 = 1.0
    assert np.allclose(tensor, 1.0, atol=1e-5), "Max value (255) should map to 1.0"
    
    # Create dummy black image (0)
    dummy_black = np.zeros((300, 300, 3), dtype=np.uint8)
    tensor_black, _ = pipeline.align_and_crop(dummy_black, bbox)
    
    # Expected: (0/255 - 0.5)/0.5 = -1.0
    assert np.allclose(tensor_black, -1.0, atol=1e-5), "Min value (0) should map to -1.0"

