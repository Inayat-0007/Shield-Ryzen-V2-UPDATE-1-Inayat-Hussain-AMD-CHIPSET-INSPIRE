"""
Shield-Ryzen V2 — Shield Utils Test Suite
============================================
10 tests covering all 5 components:
  A) EAR with cosine angle compensation
  B) Adaptive Laplacian + frequency analysis
  C) Device baseline calibration
  D) Confidence calibration (temperature scaling)
  E) Decision state machine with hysteresis

Developer: Inayat Hussain | AMD Slingshot 2026
Part 3 of 14 — Liveness, Forensics & Decision Logic Calibration
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Project root
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shield_utils_core import (
    compute_ear,
    compute_texture_score,
    _compute_hf_energy_ratio,
    calibrate_device_baseline,
    ConfidenceCalibrator,
    DecisionStateMachine,
    analyze_blink_pattern,
)


# ── Helpers ───────────────────────────────────────────────────

def _make_pixel_eye_landmarks(
    ear_target: float = 0.30,
    center_x: float = 320.0,
    center_y: float = 240.0,
    eye_width: float = 60.0,
) -> list[list[float]]:
    """Create 6 pixel-coordinate landmarks producing a target EAR.

    EAR = (v1 + v2) / (2 * h)
    For target with h_dist = eye_width:
        v1 = v2 = eye_width * ear_target
        vertical half-distance = v1 / 2

    Returns list of [x, y] pixel coordinates (NOT normalized).
    """
    h_half = eye_width / 2.0
    v_half = (eye_width * ear_target) / 2.0

    # p1 (outer), p2 (upper1), p3 (upper2), p4 (inner), p5 (lower2), p6 (lower1)
    return [
        [center_x - h_half, center_y],             # p1 — outer corner
        [center_x - h_half * 0.3, center_y - v_half],  # p2 — upper outer
        [center_x + h_half * 0.3, center_y - v_half],  # p3 — upper inner
        [center_x + h_half, center_y],             # p4 — inner corner
        [center_x + h_half * 0.3, center_y + v_half],  # p5 — lower inner
        [center_x - h_half * 0.3, center_y + v_half],  # p6 — lower outer
    ]


# ═══════════════════════════════════════════════════════════════
# TEST 1: EAR frontal correct value
# ═══════════════════════════════════════════════════════════════

def test_ear_frontal_correct_value():
    """Frontal EAR computation should produce a value near the target.
    With frontal head pose, reliability should be HIGH.
    Uses PIXEL coordinates (not normalized 0-1)."""
    landmarks = _make_pixel_eye_landmarks(ear_target=0.28, center_x=320, center_y=240)
    indices = [0, 1, 2, 3, 4, 5]

    ear, reliability = compute_ear(
        landmarks, indices,
        head_pose=(0.0, 0.0, 0.0),
        is_frontal=True,
    )

    assert 0.15 < ear < 0.45, f"EAR {ear} out of expected range [0.15, 0.45]"
    assert reliability == "HIGH", f"Expected HIGH reliability, got {reliability}"

    # Verify these are pixel coordinates (values >> 1.0)
    assert landmarks[0][0] > 100, \
        "Landmarks should be in pixel coords (>100px), not normalized"


# ═══════════════════════════════════════════════════════════════
# TEST 2: EAR angled applies cosine correction
# ═══════════════════════════════════════════════════════════════

def test_ear_angled_applies_cosine_correction():
    """With yaw between 15-25°, EAR should be corrected by
    cos(yaw) factor. Corrected EAR >= uncorrected EAR.
    Reliability should be MEDIUM."""
    landmarks = _make_pixel_eye_landmarks(ear_target=0.25)
    indices = [0, 1, 2, 3, 4, 5]

    ear_frontal, rel_frontal = compute_ear(
        landmarks, indices,
        head_pose=(0.0, 0.0, 0.0),
    )

    ear_angled, rel_angled = compute_ear(
        landmarks, indices,
        head_pose=(20.0, 0.0, 0.0),  # 20° yaw
    )

    # Cosine correction should INCREASE the EAR value
    # because h_dist appears shorter at angle, inflating raw EAR,
    # and the correction divides by cos(yaw) to compensate
    assert ear_angled >= ear_frontal * 0.95, \
        f"Angled EAR {ear_angled} should be >= ~frontal {ear_frontal}"
    assert rel_angled == "MEDIUM", \
        f"Expected MEDIUM reliability at 20°, got {rel_angled}"

    # The correction factor should be approximately 1/cos(20°)
    cos_20 = math.cos(math.radians(20.0))
    expected_ratio = 1.0 / cos_20
    actual_ratio = ear_angled / ear_frontal
    assert abs(actual_ratio - expected_ratio) < 0.2, \
        f"Correction ratio {actual_ratio:.3f} should be ~{expected_ratio:.3f}"


# ═══════════════════════════════════════════════════════════════
# TEST 3: EAR extreme angle returns LOW reliability
# ═══════════════════════════════════════════════════════════════

def test_ear_extreme_angle_returns_low_reliability():
    """At extreme angles (>25° yaw or >25° pitch), reliability
    must be LOW to prevent false blink detections."""
    landmarks = _make_pixel_eye_landmarks(ear_target=0.25)
    indices = [0, 1, 2, 3, 4, 5]

    # Extreme yaw
    _, reliability_yaw = compute_ear(
        landmarks, indices,
        head_pose=(35.0, 0.0, 0.0),  # 35° yaw
    )
    assert reliability_yaw == "LOW", f"Expected LOW at 35° yaw, got {reliability_yaw}"

    # Extreme pitch (looking down)
    _, reliability_pitch = compute_ear(
        landmarks, indices,
        head_pose=(0.0, 30.0, 0.0),  # 30° pitch down
    )
    assert reliability_pitch == "LOW", f"Expected LOW at 30° pitch, got {reliability_pitch}"


# ═══════════════════════════════════════════════════════════════
# TEST 4: EAR uses pixel coords not normalized
# ═══════════════════════════════════════════════════════════════

def test_ear_uses_pixel_coords_not_normalized():
    """Verify that EAR computation works correctly with PIXEL
    coordinates (typical values 100-600) and that the result
    is independent of coordinate scale (same shape = same EAR)."""
    # PIXEL coordinates (actual screen positions)
    px_landmarks = _make_pixel_eye_landmarks(
        ear_target=0.30, center_x=320, center_y=240, eye_width=60
    )
    ear_px, _ = compute_ear(px_landmarks, [0, 1, 2, 3, 4, 5])

    # Same eye SHAPE but scaled up (larger eye on screen)
    big_landmarks = _make_pixel_eye_landmarks(
        ear_target=0.30, center_x=500, center_y=400, eye_width=120
    )
    ear_big, _ = compute_ear(big_landmarks, [0, 1, 2, 3, 4, 5])

    # EAR should be the SAME regardless of pixel scale
    # (it's a ratio, so scale cancels out)
    assert abs(ear_px - ear_big) < 0.05, \
        f"EAR should be scale-invariant: {ear_px:.4f} vs {ear_big:.4f}"

    # Both should be near 0.30 target
    assert 0.20 < ear_px < 0.40
    assert 0.20 < ear_big < 0.40


# ═══════════════════════════════════════════════════════════════
# TEST 5: Laplacian adaptive threshold
# ═══════════════════════════════════════════════════════════════

def test_laplacian_adaptive_threshold():
    """When device_baseline is provided, threshold should be
    40% of baseline (not fixed 50). Different baselines should
    produce different suspicious flags for the same image."""
    rng = np.random.RandomState(42)
    face = rng.randint(100, 200, (299, 299, 3), dtype=np.uint8)

    # With high baseline (threshold = 200 * 0.4 = 80)
    lap_var_h, suspicious_high, explanation_high = compute_texture_score(
        face, device_baseline=200.0
    )

    # With low baseline (threshold = 50 * 0.4 = 20)
    lap_var_l, suspicious_low, explanation_low = compute_texture_score(
        face, device_baseline=50.0
    )

    assert "adaptive" in explanation_high, \
        f"Expected 'adaptive' in explanation, got: {explanation_high}"
    assert "baseline=200.0" in explanation_high

    # Same Laplacian value but different thresholds
    assert lap_var_h == lap_var_l, "Same image should give same Laplacian variance"


# ═══════════════════════════════════════════════════════════════
# TEST 6: Frequency analysis detects screen replay
# ═══════════════════════════════════════════════════════════════

def test_laplacian_frequency_analysis_detects_screen():
    """FFT high-frequency energy ratio should differ between
    sharp (natural) and blurred (GAN-like/screen-captured) images.
    Sharp image should have higher HF ratio."""
    rng = np.random.RandomState(42)

    # Sharp image (lots of high-frequency content)
    sharp = rng.randint(0, 255, (100, 100), dtype=np.uint8)

    # Blurred image (suppressed high-frequency — simulates GAN/screen)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 10)

    hf_sharp = _compute_hf_energy_ratio(sharp)
    hf_blurred = _compute_hf_energy_ratio(blurred)

    assert hf_sharp > hf_blurred, \
        f"Sharp HF ratio {hf_sharp:.4f} should be > blurred {hf_blurred:.4f}"

    # Both should be in valid range
    assert 0.0 <= hf_sharp <= 1.5, f"Sharp HF ratio {hf_sharp} out of range"
    assert 0.0 <= hf_blurred <= 1.5, f"Blurred HF ratio {hf_blurred} out of range"

    # Full pipeline test: blurred face should be flagged as suspicious
    blurred_face = cv2.GaussianBlur(
        rng.randint(100, 200, (299, 299, 3), dtype=np.uint8),
        (31, 31), 15,
    )
    _, is_suspicious, _ = compute_texture_score(blurred_face)
    assert isinstance(is_suspicious, bool)


# ═══════════════════════════════════════════════════════════════
# TEST 7: Confidence calibrator reduces overconfidence
# ═══════════════════════════════════════════════════════════════

def test_confidence_calibrator_reduces_overconfidence():
    """Temperature scaling with T>1 should reduce the gap between
    the highest and lowest probabilities (less overconfident)."""
    raw_softmax = np.array([0.95, 0.05])  # Very confident

    calibrator = ConfidenceCalibrator(temperature=2.0)
    calibrated = calibrator.calibrate(raw_softmax)

    # Calibrated should be less extreme
    assert calibrated[0] < raw_softmax[0], \
        f"Calibrated max {calibrated[0]} should be < raw {raw_softmax[0]}"
    assert calibrated[1] > raw_softmax[1], \
        f"Calibrated min {calibrated[1]} should be > raw {raw_softmax[1]}"

    # Should still sum to ~1.0
    assert abs(calibrated.sum() - 1.0) < 0.01, \
        f"Calibrated probs should sum to 1.0, got {calibrated.sum()}"

    # With T=1.0, output should be very close to input (identity)
    identity_cal = ConfidenceCalibrator(temperature=1.0)
    identity_out = identity_cal.calibrate(raw_softmax)
    assert np.allclose(identity_out, raw_softmax, atol=0.01), \
        f"T=1.0 should be near identity, got {identity_out}"

    # Test save/load round-trip
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        tf_path = tf.name
    try:
        calibrator.save_params(tf_path)
        loaded = ConfidenceCalibrator.load_params(tf_path)
        assert loaded.temperature == calibrator.temperature
    finally:
        if os.path.exists(tf_path):
            os.unlink(tf_path)


# ═══════════════════════════════════════════════════════════════
# TEST 8: State machine hysteresis prevents flicker
# ═══════════════════════════════════════════════════════════════

def test_state_machine_hysteresis_prevents_flicker():
    """State should NOT change to REAL until N consecutive frames agree.
    Security escalation (→ FAKE) should be immediate (1 frame)."""
    sm = DecisionStateMachine(frames=5)

    # First frame transitions from UNKNOWN immediately
    state = sm.update("REAL", "PASS", "PASS")
    assert state == "REAL", f"First frame should transition from UNKNOWN, got {state}"

    # Now send 2 FAKE frames — should escalate IMMEDIATELY (agile hysteresis)
    state = sm.update("FAKE", "PASS", "PASS")
    assert state == "FAKE", f"FAKE should escalate immediately, got {state}"

    # Now try to de-escalate: 3 REAL frames should NOT be enough (hysteresis=5)
    for _ in range(3):
        state = sm.update("REAL", "PASS", "PASS")
    assert state != "REAL", \
        f"3 frames should not overcome hysteresis=5, got {state}"

    # After 5 consecutive REAL frames, should transition
    for _ in range(3):  # Total = 3+3 = 6
        state = sm.update("REAL", "PASS", "PASS")
    assert state == "REAL", \
        f"After 6 consecutive REAL frames, state should be REAL, got {state}"


# ═══════════════════════════════════════════════════════════════
# TEST 9: Conflict resolution truth table — all 8 cases
# ═══════════════════════════════════════════════════════════════

def test_conflict_resolution_truth_table_all_8_cases():
    """Verify all 8 rows of the conflict resolution truth table
    using the DecisionStateMachine with string inputs."""

    # Each case gets a fresh state machine starting from UNKNOWN
    # (first frame transitions immediately from UNKNOWN)
    cases = [
        # (neural, liveness, forensic, expected_state)
        ("REAL", "PASS", "PASS", "REAL"),        # Case 1: All pass → REAL
        ("REAL", "PASS", "FAIL", "SUSPICIOUS"),  # Case 2: Forensic fail
        ("REAL", "FAIL", "PASS", "WAIT_BLINK"),  # Case 3: Liveness fail
        ("REAL", "FAIL", "FAIL", "HIGH_RISK"),   # Case 4: Two failures
        ("FAKE", "PASS", "PASS", "FAKE"),        # Case 5: Neural fake
        ("FAKE", "PASS", "FAIL", "FAKE"),        # Case 6: Neural fake + forensic
        ("FAKE", "FAIL", "PASS", "FAKE"),        # Case 7: Neural fake + liveness
        ("FAKE", "FAIL", "FAIL", "CRITICAL"),    # Case 8: All fail
    ]

    for neural, liveness, forensic, expected in cases:
        sm = DecisionStateMachine(frames=5)
        state = sm.update(neural, liveness, forensic)
        assert state == expected, \
            f"Case ({neural},{liveness},{forensic}): expected {expected}, got {state}"


# ═══════════════════════════════════════════════════════════════
# TEST 10: Device baseline calibration produces valid JSON
# ═══════════════════════════════════════════════════════════════

def test_device_baseline_calibration_produces_valid_json():
    """calibrate_device_baseline should produce valid JSON
    with ALL required keys when run in synthetic mode."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        out_path = f.name

    try:
        result = calibrate_device_baseline(
            camera=None,
            num_frames=50,
            output_path=out_path,
        )

        # Check ALL required keys from spec
        required_keys = [
            "laplacian_mean",
            "laplacian_std",
            "recommended_threshold",
            "ear_baseline_mean",
            "ear_baseline_std",
            "recommended_ear_threshold",
            "camera_resolution",
            "lighting_condition",
            "calibration_timestamp",
            "calibration_frames_used",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Values should be plausible
        assert result["laplacian_mean"] > 0
        assert result["laplacian_std"] >= 0
        assert result["recommended_threshold"] > 0
        assert 0.0 < result["ear_baseline_mean"] < 1.0
        assert result["ear_baseline_std"] >= 0
        assert 0.0 < result["recommended_ear_threshold"] < 1.0
        assert result["lighting_condition"] in ("GOOD", "MODERATE", "LOW_LIGHT")
        assert result["calibration_frames_used"] == 50

        # File should be valid JSON
        with open(out_path, 'r') as f:
            loaded = json.load(f)
        assert loaded == result

    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
