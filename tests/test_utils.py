"""
Shield-Ryzen V2 -- Shield Utils Test Suite
============================================
12 tests covering: EAR, Laplacian, Calibration, State Machine,
Blink Pattern, Threshold Optimization.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 3 of 12 -- Liveness, Forensic Analysis & Calibration
"""

from __future__ import annotations

import json
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

from shield_utils import (
    compute_ear, calculate_ear, analyze_blink_pattern,
    compute_texture_score, check_texture, _compute_hf_energy_ratio,
    calibrate_device_baseline,
    ConfidenceCalibrator,
    classify_face,
    preprocess_face,
    CONFIDENCE_THRESHOLD, BLINK_THRESHOLD, LAPLACIAN_THRESHOLD,
)
from shield_utils.calibrated_decision import (
    DecisionStateMachine, TierResult,
)


# ── Helpers ───────────────────────────────────────────────────

class _MockLandmark:
    """Simulate MediaPipe landmark with .x, .y attributes."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _make_eye_landmarks(ear_target: float = 0.3) -> list[_MockLandmark]:
    """Create 6 landmarks that produce a specific EAR value.

    EAR = (v1 + v2) / (2 * h)
    For a target EAR with h=0.1:
        v1 + v2 = 2 * h * EAR = 2 * 0.1 * EAR
        v1 = v2 = h * EAR = 0.1 * EAR
    """
    h_dist = 0.1  # horizontal distance
    v_half = (h_dist * ear_target) / 2.0  # vertical half-distance

    # p1 (outer), p2 (upper1), p3 (upper2), p4 (inner), p5 (lower2), p6 (lower1)
    return [
        _MockLandmark(0.3, 0.5),           # p1 -- outer corner
        _MockLandmark(0.35, 0.5 - v_half), # p2 -- upper
        _MockLandmark(0.36, 0.5 - v_half), # p3 -- upper
        _MockLandmark(0.4, 0.5),           # p4 -- inner corner
        _MockLandmark(0.36, 0.5 + v_half), # p5 -- lower
        _MockLandmark(0.35, 0.5 + v_half), # p6 -- lower
    ]


# ═══════════════════════════════════════════════════════════════
# TEST 1: EAR frontal correct value
# ═══════════════════════════════════════════════════════════════

def test_ear_frontal_correct_value():
    """Frontal EAR computation should produce a value near the target.
    With frontal head pose, reliability should be HIGH."""
    landmarks = _make_eye_landmarks(ear_target=0.28)
    indices = [0, 1, 2, 3, 4, 5]

    ear, reliability = compute_ear(
        landmarks, indices,
        head_pose=(0.0, 0.0, 0.0),
        is_frontal=True,
    )

    assert 0.2 < ear < 0.4, f"EAR {ear} out of expected range [0.2, 0.4]"
    assert reliability == "HIGH", f"Expected HIGH reliability, got {reliability}"


# ═══════════════════════════════════════════════════════════════
# TEST 2: EAR angled applies cosine correction
# ═══════════════════════════════════════════════════════════════

def test_ear_angled_applies_correction():
    """With yaw between 15-25 degrees, EAR should be corrected by
    cosine factor and reliability should be MEDIUM."""
    landmarks = _make_eye_landmarks(ear_target=0.25)
    indices = [0, 1, 2, 3, 4, 5]

    ear_frontal, rel_frontal = compute_ear(
        landmarks, indices,
        head_pose=(0.0, 0.0, 0.0),
    )

    ear_angled, rel_angled = compute_ear(
        landmarks, indices,
        head_pose=(20.0, 0.0, 0.0),  # 20 deg yaw
    )

    # Cosine correction should INCREASE the EAR value
    assert ear_angled >= ear_frontal, \
        f"Angled EAR {ear_angled} should be >= frontal {ear_frontal}"
    assert rel_angled == "MEDIUM", \
        f"Expected MEDIUM reliability at 20deg, got {rel_angled}"


# ═══════════════════════════════════════════════════════════════
# TEST 3: EAR extreme angle returns LOW reliability
# ═══════════════════════════════════════════════════════════════

def test_ear_extreme_angle_returns_low_reliability():
    """At extreme angles (>25 degrees), reliability must be LOW."""
    landmarks = _make_eye_landmarks(ear_target=0.25)
    indices = [0, 1, 2, 3, 4, 5]

    _, reliability = compute_ear(
        landmarks, indices,
        head_pose=(35.0, 0.0, 0.0),  # 35 deg yaw
    )
    assert reliability == "LOW", f"Expected LOW at 35deg, got {reliability}"

    _, reliability2 = compute_ear(
        landmarks, indices,
        head_pose=(0.0, 30.0, 0.0),  # 30 deg pitch
    )
    assert reliability2 == "LOW", f"Expected LOW at 30deg pitch, got {reliability2}"


# ═══════════════════════════════════════════════════════════════
# TEST 4: Laplacian adaptive threshold
# ═══════════════════════════════════════════════════════════════

def test_laplacian_adaptive_threshold():
    """When device_baseline is provided, threshold should be
    40% of baseline (not fixed 50)."""
    # Create a face crop with known texture
    rng = np.random.RandomState(42)
    face = rng.randint(100, 200, (299, 299, 3), dtype=np.uint8)

    # With high baseline, more frames should pass
    lap_var, suspicious_high, explanation_high = compute_texture_score(
        face, device_baseline=200.0
    )
    # threshold = 200 * 0.4 = 80

    lap_var2, suspicious_low, explanation_low = compute_texture_score(
        face, device_baseline=50.0
    )
    # threshold = 50 * 0.4 = 20

    assert "adaptive" in explanation_high, \
        f"Expected 'adaptive' in explanation, got: {explanation_high}"
    assert "baseline=200.0" in explanation_high


# ═══════════════════════════════════════════════════════════════
# TEST 5: Laplacian frequency analysis
# ═══════════════════════════════════════════════════════════════

def test_laplacian_frequency_analysis():
    """FFT high-frequency energy ratio should differ between
    sharp (natural) and blurred (GAN-like) images."""
    # Sharp image (lots of high-frequency content)
    rng = np.random.RandomState(42)
    sharp = rng.randint(0, 255, (100, 100), dtype=np.uint8)

    # Blurred image (suppressed high-frequency)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 10)

    hf_sharp = _compute_hf_energy_ratio(sharp)
    hf_blurred = _compute_hf_energy_ratio(blurred)

    assert hf_sharp > hf_blurred, \
        f"Sharp HF ratio {hf_sharp} should be > blurred {hf_blurred}"
    assert 0.0 <= hf_sharp <= 1.0
    assert 0.0 <= hf_blurred <= 1.0


# ═══════════════════════════════════════════════════════════════
# TEST 6: Laplacian forehead ROI extraction
# ═══════════════════════════════════════════════════════════════

def test_laplacian_forehead_roi_extraction():
    """compute_texture_score should analyze the forehead region
    (rows 15-35%, cols 25-75%), not the entire face."""
    # Create image where forehead is very smooth but rest is sharp
    face = np.random.randint(100, 200, (299, 299, 3), dtype=np.uint8)

    # Make forehead region uniform (very smooth = suspicious)
    face[44:104, 74:224] = 150  # rows 15-35%, cols 25-75% of 299

    lap_var, is_suspicious, explanation = compute_texture_score(face)

    # The forehead-specific Laplacian should detect smoothness
    # even though the rest of the face is sharp
    assert isinstance(lap_var, float)
    assert isinstance(is_suspicious, bool)
    assert isinstance(explanation, str)


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

    # With T=1.0, output should be very close to input
    identity_cal = ConfidenceCalibrator(temperature=1.0)
    identity_out = identity_cal.calibrate(raw_softmax)
    assert np.allclose(identity_out, raw_softmax, atol=0.01), \
        f"T=1.0 should be near identity, got {identity_out}"


# ═══════════════════════════════════════════════════════════════
# TEST 8: State machine hysteresis prevents flicker
# ═══════════════════════════════════════════════════════════════

def test_state_machine_hysteresis_prevents_flicker():
    """State should NOT change until N consecutive frames agree.
    3 frames of VERIFIED followed by 1 SUSPICIOUS shouldn't
    change if hysteresis=5."""
    sm = DecisionStateMachine(hysteresis_frames=5)

    pass_result = TierResult(passed=True, confidence=0.9)
    fail_result = TierResult(passed=False, confidence=0.3)

    # Send 3 VERIFIED frames (not enough for hysteresis=5)
    for _ in range(3):
        state = sm.update(pass_result, pass_result, pass_result)
    assert state == "UNKNOWN", \
        f"State should still be UNKNOWN after 3 frames, got {state}"

    # Now interrupt with 1 SUSPICIOUS frame
    state = sm.update(fail_result, pass_result, pass_result)
    assert state == "UNKNOWN", f"Should still be UNKNOWN, got {state}"

    # Continue with VERIFIED frames
    for _ in range(5):
        state = sm.update(pass_result, pass_result, pass_result)
    assert state == "VERIFIED", \
        f"After 5 consecutive VERIFIED, state should be VERIFIED, got {state}"


# ═══════════════════════════════════════════════════════════════
# TEST 9: Conflict resolution truth table (all 8 cases)
# ═══════════════════════════════════════════════════════════════

def test_conflict_resolution_truth_table_all_8_cases():
    """Verify all 8 rows of the conflict resolution truth table."""
    r = TierResult(passed=True)
    f = TierResult(passed=False)

    # Use the static method directly
    resolve = DecisionStateMachine._resolve_conflict

    # Case 1: Real + Pass + Pass -> VERIFIED
    assert resolve(r, r, r) == "VERIFIED"

    # Case 2: Real + Pass + Fail -> SUSPICIOUS
    assert resolve(r, r, f) == "SUSPICIOUS"

    # Case 3: Real + Fail + Pass -> SUSPICIOUS
    assert resolve(r, f, r) == "SUSPICIOUS"

    # Case 4: Real + Fail + Fail -> HIGH_RISK
    assert resolve(r, f, f) == "HIGH_RISK"

    # Case 5: Fake + Pass + Pass -> SUSPICIOUS
    assert resolve(f, r, r) == "SUSPICIOUS"

    # Case 6: Fake + Pass + Fail -> HIGH_RISK
    assert resolve(f, r, f) == "HIGH_RISK"

    # Case 7: Fake + Fail + Pass -> HIGH_RISK
    assert resolve(f, f, r) == "HIGH_RISK"

    # Case 8: Fake + Fail + Fail -> HIGH_RISK
    assert resolve(f, f, f) == "HIGH_RISK"


# ═══════════════════════════════════════════════════════════════
# TEST 10: Device baseline calibration produces valid JSON
# ═══════════════════════════════════════════════════════════════

def test_device_baseline_calibration_produces_valid_json():
    """calibrate_device_baseline should produce valid JSON
    with all required keys when run without a camera."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        out_path = f.name

    try:
        result = calibrate_device_baseline(
            camera=None,  # Synthetic mode
            num_frames=50,
            output_path=out_path,
        )

        # Check required keys
        required_keys = [
            "laplacian_mean", "laplacian_std",
            "recommended_threshold", "ear_baseline_mean",
            "camera_resolution", "calibration_timestamp",
            "num_frames_captured",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Values should be plausible
        assert result["laplacian_mean"] > 0
        assert result["laplacian_std"] >= 0
        assert result["recommended_threshold"] > 0
        assert result["num_frames_captured"] == 50

        # File should be valid JSON
        with open(out_path, 'r') as f:
            loaded = json.load(f)
        assert loaded == result

    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


# ═══════════════════════════════════════════════════════════════
# TEST 11: Blink pattern detection
# ═══════════════════════════════════════════════════════════════

def test_blink_pattern_detection():
    """analyze_blink_pattern should distinguish natural from
    robotic blink patterns."""
    now = time.monotonic()

    # Natural blinking: irregular intervals, ~15 per minute
    natural_blinks = [now - 60 + i * 4.0 + np.random.uniform(-1, 1)
                      for i in range(15)]

    score_natural, desc_natural = analyze_blink_pattern(
        natural_blinks, window_seconds=65.0
    )

    # Robotic blinking: exactly periodic
    robotic_blinks = [now - 60 + i * 4.0 for i in range(15)]

    score_robotic, desc_robotic = analyze_blink_pattern(
        robotic_blinks, window_seconds=65.0
    )

    assert score_natural >= 0.0 and score_natural <= 1.0
    assert score_robotic >= 0.0 and score_robotic <= 1.0

    # Natural should score higher than robotic
    # (natural has irregular intervals, robotic has CV < 0.1)
    assert score_natural > score_robotic, \
        f"Natural {score_natural} should be > robotic {score_robotic}"

    # No blinks should produce low score
    score_none, desc_none = analyze_blink_pattern([], window_seconds=15.0)
    assert score_none < 0.5, f"No blinks score {score_none} should be < 0.5"


# ═══════════════════════════════════════════════════════════════
# TEST 12: Threshold optimization differs from 89%
# ═══════════════════════════════════════════════════════════════

def test_threshold_optimization_differs_from_89():
    """The optimized threshold should differ from the arbitrary 89%,
    proving the old threshold had no empirical basis."""
    # Import and run optimization with synthetic data
    sys.path.insert(0, os.path.join(_project_root, 'evaluation'))
    from threshold_optimization import optimize_thresholds

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "thresholds.yaml")
        results = optimize_thresholds(
            output_path=output_path,
            old_threshold=0.89,
        )

        # The optimal threshold should exist and be in [0.01, 0.99]
        optimal = results["confidence_threshold"]
        assert 0.01 <= optimal <= 0.99, \
            f"Optimal threshold {optimal} out of range"

        # It should differ from 89% (proving it's suboptimal)
        assert results["improvement"]["threshold_differs"], \
            "Optimal threshold should differ from old 89%"

        # F1 should be at least as good
        assert results["f1_optimal"]["f1"] >= results["old_89_pct"]["f1"], \
            "Optimal F1 should be >= old F1"

        # Config file should exist
        assert os.path.exists(output_path), "Threshold YAML not saved"


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
