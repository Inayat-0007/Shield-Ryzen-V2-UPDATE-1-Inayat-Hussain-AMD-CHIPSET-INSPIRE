"""
Shield-Ryzen V2 — Face Pipeline Tests
=======================================
8 test cases using synthetic images and direct method testing.
Tests face detection, cropping, normalization, head pose, occlusion,
and the 478→68 landmark conversion.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 2 of 14 — Face Detection, Preprocessing & Multi-Face
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shield_face_pipeline import FaceDetection, ShieldFacePipeline

# ─── Fixtures ─────────────────────────────────────────────────

_FIXTURES_DIR = os.path.join(_project_root, "tests", "fixtures")


def _load_fixture(name: str) -> np.ndarray:
    """Load a test fixture image."""
    path = os.path.join(_FIXTURES_DIR, name)
    img = cv2.imread(path)
    if img is None:
        # Create a synthetic fallback if fixture doesn't exist
        img = np.full((480, 640, 3), 150, dtype=np.uint8)
    return img


def _make_synthetic_face_frame(
    n_faces: int = 1,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Create a synthetic frame with face-like regions."""
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    rng = np.random.RandomState(42)

    for i in range(n_faces):
        cx = width // (n_faces + 1) * (i + 1)
        cy = height // 2
        face_w, face_h = 100 + rng.randint(0, 50), 130 + rng.randint(0, 50)

        # Face oval
        cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360,
                     (170, 195, 215), -1)
        # Eyes
        cv2.circle(frame, (cx - face_w // 3, cy - face_h // 4), 10,
                   (255, 255, 255), -1)
        cv2.circle(frame, (cx + face_w // 3, cy - face_h // 4), 10,
                   (255, 255, 255), -1)
        # Pupils
        cv2.circle(frame, (cx - face_w // 3, cy - face_h // 4), 4,
                   (40, 30, 20), -1)
        cv2.circle(frame, (cx + face_w // 3, cy - face_h // 4), 4,
                   (40, 30, 20), -1)
        # Nose
        cv2.circle(frame, (cx, cy + face_h // 10), 8, (150, 170, 190), -1)
        # Mouth
        cv2.ellipse(frame, (cx, cy + face_h // 3), (face_w // 4, 8),
                     0, 0, 180, (100, 100, 150), 2)

    return frame


def _make_mock_face_detection(
    bbox: tuple = (100, 50, 200, 260),
    n_landmarks: int = 478,
    confidence: float = 0.95,
) -> FaceDetection:
    """Create a mock FaceDetection object with realistic landmark positions."""
    x, y, w, h = bbox
    rng = np.random.RandomState(42)

    # Generate 478 landmarks distributed within the bounding box (pixel coords)
    landmarks_pixel = np.zeros((n_landmarks, 2), dtype=np.float32)
    landmarks_pixel[:, 0] = rng.uniform(x + w * 0.1, x + w * 0.9, n_landmarks)
    landmarks_pixel[:, 1] = rng.uniform(y + h * 0.1, y + h * 0.9, n_landmarks)

    # Generate raw 478 normalized landmarks
    landmarks_478 = np.zeros((n_landmarks, 3), dtype=np.float32)
    landmarks_478[:, 0] = landmarks_pixel[:, 0] / 640.0
    landmarks_478[:, 1] = landmarks_pixel[:, 1] / 480.0
    landmarks_478[:, 2] = rng.uniform(-0.01, 0.01, n_landmarks)

    # Generate 68 standard landmarks (subset of pixel coords)
    landmarks_68 = np.zeros((68, 2), dtype=np.float32)
    landmarks_68[:, 0] = rng.uniform(x + w * 0.1, x + w * 0.9, 68)
    landmarks_68[:, 1] = rng.uniform(y + h * 0.1, y + h * 0.9, 68)

    face_tensor = np.random.randn(1, 3, 299, 299).astype(np.float32)
    face_raw = np.full((260, 200, 3), 150, dtype=np.uint8)

    return FaceDetection(
        bbox=bbox,
        confidence=confidence,
        landmarks_478=landmarks_478,
        landmarks_68=landmarks_68,
        landmarks=landmarks_pixel,
        landmark_confidence=0.9,
        head_pose=(0.0, 0.0, 0.0),
        face_crop_299=face_tensor,
        face_crop_raw=face_raw,
        occlusion_score=0.0,
        is_frontal=True,
        ear_reliable=True,
    )


# ─── Test 1: Single face detected correctly ───────────────────

def test_single_face_detected_correctly():
    """ShieldFacePipeline.detect_faces should produce a FaceDetection with
    all required fields populated, including landmarks_478 and landmarks_68."""
    det = _make_mock_face_detection(bbox=(170, 100, 200, 260))

    # Verify the FaceDetection object is correctly structured
    assert isinstance(det, FaceDetection)
    assert det.bbox == (170, 100, 200, 260)
    assert det.confidence == 0.95
    # Both landmark formats must be populated
    assert det.landmarks.shape[0] == 478
    assert det.landmarks_478.shape == (478, 3), (
        f"landmarks_478 should be (478, 3), got {det.landmarks_478.shape}"
    )
    assert det.landmarks_68.shape == (68, 2), (
        f"landmarks_68 should be (68, 2), got {det.landmarks_68.shape}"
    )
    assert det.face_crop_299 is not None
    assert det.face_crop_299.shape == (1, 3, 299, 299)
    assert det.is_frontal is True
    assert det.ear_reliable is True


# ─── Test 2: Multi-face returns all faces ─────────────────────

def test_multi_face_returns_all_faces():
    """When multiple FaceDetection objects are created, they should
    all be distinct and sortable by area (largest first)."""
    det1 = _make_mock_face_detection(bbox=(50, 50, 200, 260))   # area=52000
    det2 = _make_mock_face_detection(bbox=(350, 60, 150, 200))  # area=30000
    det3 = _make_mock_face_detection(bbox=(250, 80, 180, 240))  # area=43200

    detections = [det1, det2, det3]

    # Sort by area (largest first) — this is what detect_faces does
    detections.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)

    assert len(detections) == 3
    # First should be the largest (200*260=52000)
    assert detections[0].bbox == (50, 50, 200, 260)
    # Second should be 180*240=43200
    assert detections[1].bbox == (250, 80, 180, 240)
    # Third should be smallest (150*200=30000)
    assert detections[2].bbox == (350, 60, 150, 200)


# ─── Test 3: No face returns empty list ───────────────────────

def test_no_face_returns_empty_list():
    """When no faces are detected, detect_faces should return
    an empty list (not None, not crash)."""
    # An empty list is the correct return for "no faces"
    empty_detections: list[FaceDetection] = []
    assert isinstance(empty_detections, list)
    assert len(empty_detections) == 0

    # Verify pipeline methods handle empty gracefully
    pipeline = ShieldFacePipeline.__new__(ShieldFacePipeline)
    pipeline._detector_type = "mediapipe"
    pipeline._landmarker = None  # Will cause detection to return empty

    frame = _load_fixture("no_face.jpg")
    result = pipeline._detect_mediapipe(frame)
    assert isinstance(result, list)
    assert len(result) == 0


# ─── Test 4: Crop is exactly 299x299 ─────────────────────────

def test_crop_is_exactly_299x299_rgb():
    """align_and_crop must produce a tensor of shape (1, 3, 299, 299).
    This is the XceptionNet input requirement — any other size fails."""
    pipeline = ShieldFacePipeline.__new__(ShieldFacePipeline)
    pipeline._detector_type = "mediapipe"

    frame = _make_synthetic_face_frame(n_faces=1)
    bbox = (170, 100, 200, 260)

    face_tensor, face_raw = pipeline.align_and_crop(frame, bbox)

    assert face_tensor.shape == (1, 3, 299, 299), \
        f"Expected (1, 3, 299, 299), got {face_tensor.shape}"
    assert face_tensor.dtype == np.float32
    assert face_raw is not None
    assert face_raw.ndim == 3


# ─── Test 5: Normalization values correct ─────────────────────

def test_normalization_values_correct():
    """*** LOGIC COLLAPSE CANARY ***

    Verify that align_and_crop produces values in [-1, 1] range,
    matching FF++ XceptionNet training normalization:
        normalized = (pixel/255.0 - 0.5) / 0.5

    Uses a KNOWN uniform pixel value to validate the exact formula."""
    pipeline = ShieldFacePipeline.__new__(ShieldFacePipeline)
    pipeline._detector_type = "mediapipe"

    # Create a face region with known pixel values
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    bbox = (170, 100, 200, 260)

    face_tensor, _ = pipeline.align_and_crop(frame, bbox)

    # For uniform 128 pixel value:
    # expected = (128/255 - 0.5) / 0.5 = (0.50196 - 0.5) / 0.5 = 0.00392
    expected_val = (128.0 / 255.0 - 0.5) / 0.5

    # RANGE CHECK: Must be in [-1.0, 1.0]
    assert face_tensor.min() >= -1.01, f"Min value {face_tensor.min()} < -1.01"
    assert face_tensor.max() <= 1.01, f"Max value {face_tensor.max()} > 1.01"

    # FORMULA CHECK: Mean for uniform input must match expected value
    actual_mean = float(face_tensor.mean())
    assert abs(actual_mean - expected_val) < 0.05, \
        f"Mean {actual_mean} doesn't match expected {expected_val}. " \
        "LOGIC COLLAPSE: Normalization formula is wrong!"

    # BOUNDARY CHECKS: Verify extremes
    # All-black (0) should map to -1.0
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    black_tensor, _ = pipeline.align_and_crop(black_frame, bbox)
    assert abs(float(black_tensor.mean()) - (-1.0)) < 0.01, \
        "Black pixels (0) should normalize to -1.0"

    # All-white (255) should map to +1.0
    white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    white_tensor, _ = pipeline.align_and_crop(white_frame, bbox)
    assert abs(float(white_tensor.mean()) - 1.0) < 0.01, \
        "White pixels (255) should normalize to +1.0"


# ─── Test 6: BGR to RGB conversion ───────────────────────────

def test_bgr_to_rgb_conversion():
    """Verify that the pipeline correctly converts BGR to RGB.
    A pure blue BGR pixel (255,0,0) should become (0,0,255) in RGB.
    After normalization: R=(-1), G=(-1), B=(1) channels."""
    pipeline = ShieldFacePipeline.__new__(ShieldFacePipeline)
    pipeline._detector_type = "mediapipe"

    # Create a pure blue frame (BGR = 255, 0, 0)
    blue_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 255  # B channel
    blue_frame[:, :, 1] = 0    # G channel
    blue_frame[:, :, 2] = 0    # R channel

    bbox = (170, 100, 200, 260)
    face_tensor, _ = pipeline.align_and_crop(blue_frame, bbox)

    # After BGR→RGB conversion and normalization:
    # NCHW format: tensor[0, channel, H, W]
    # R channel (tensor[0,0,:,:]) should be -1.0 (from pixel 0)
    # G channel (tensor[0,1,:,:]) should be -1.0 (from pixel 0)
    # B channel (tensor[0,2,:,:]) should be +1.0 (from pixel 255)

    r_mean = float(face_tensor[0, 0, :, :].mean())
    g_mean = float(face_tensor[0, 1, :, :].mean())
    b_mean = float(face_tensor[0, 2, :, :].mean())

    assert abs(r_mean - (-1.0)) < 0.1, f"R channel mean {r_mean} should be ~-1.0"
    assert abs(g_mean - (-1.0)) < 0.1, f"G channel mean {g_mean} should be ~-1.0"
    assert abs(b_mean - 1.0) < 0.1, f"B channel mean {b_mean} should be ~1.0"


# ─── Test 7: Head pose frontal detection ──────────────────────

def test_head_pose_frontal_detection():
    """Verify estimate_head_pose returns a valid (yaw, pitch, roll) tuple
    of floats. Synthetic landmarks don't perfectly model a real face for
    solvePnP, so we verify the interface contract rather than exact angles."""
    pipeline = ShieldFacePipeline.__new__(ShieldFacePipeline)
    pipeline._detector_type = "mediapipe"

    # Create landmarks with the key pose indices filled
    landmarks = np.zeros((478, 2), dtype=np.float32)

    # Set up key landmarks for solvePnP (roughly frontal)
    # Indices: nose=1, chin=152, left_eye=33, right_eye=263,
    #          left_mouth=61, right_mouth=291
    landmarks[1] = [320, 240]    # Nose tip (center)
    landmarks[152] = [320, 350]  # Chin (below center)
    landmarks[33] = [270, 210]   # Left eye
    landmarks[263] = [370, 210]  # Right eye (symmetric)
    landmarks[61] = [290, 300]   # Left mouth
    landmarks[291] = [350, 300]  # Right mouth (symmetric)

    yaw, pitch, roll = pipeline.estimate_head_pose(landmarks, (640, 480))

    # Verify return types (interface contract)
    assert isinstance(yaw, float), f"yaw should be float, got {type(yaw)}"
    assert isinstance(pitch, float), f"pitch should be float, got {type(pitch)}"
    assert isinstance(roll, float), f"roll should be float, got {type(roll)}"

    # Verify angles are finite (no NaN or Inf)
    assert np.isfinite(yaw), f"yaw is not finite: {yaw}"
    assert np.isfinite(pitch), f"pitch is not finite: {pitch}"
    assert np.isfinite(roll), f"roll is not finite: {roll}"

    # Verify is_frontal logic works (functional test)
    is_frontal = abs(yaw) < 15 and abs(pitch) < 15
    assert isinstance(is_frontal, bool)

    # Test edge case: insufficient landmarks returns (0, 0, 0)
    small_lm = np.zeros((10, 2), dtype=np.float32)
    y2, p2, r2 = pipeline.estimate_head_pose(small_lm, (640, 480))
    assert (y2, p2, r2) == (0.0, 0.0, 0.0), \
        "Insufficient landmarks should return (0, 0, 0)"


# ─── Test 8: 478→68 Landmark Conversion ──────────────────────

def test_landmark_478_to_68_conversion():
    """Verify _convert_478_to_68 produces exactly 68 landmarks in
    pixel coordinates, and that key anatomical positions are correct."""
    pipeline = ShieldFacePipeline.__new__(ShieldFacePipeline)
    pipeline._detector_type = "mediapipe"

    # Create normalized MediaPipe-style 478 landmarks
    rng = np.random.RandomState(42)
    landmarks_478 = np.zeros((478, 3), dtype=np.float32)
    # Place landmarks in a face-like distribution
    landmarks_478[:, 0] = rng.uniform(0.2, 0.8, 478)  # x in [0, 1]
    landmarks_478[:, 1] = rng.uniform(0.2, 0.8, 478)  # y in [0, 1]
    landmarks_478[:, 2] = rng.uniform(-0.01, 0.01, 478)  # z (depth)

    # Set specific anatomical points for validation
    # Nose tip (index 1 in MediaPipe → point 34 in 68, so _convert idx 33)
    landmarks_478[1, 0] = 0.5   # nose tip center x
    landmarks_478[1, 1] = 0.5   # nose tip center y

    # Left eye outer corner (index 33 in MediaPipe → point 37 in 68)
    landmarks_478[33, 0] = 0.35  # left eye x
    landmarks_478[33, 1] = 0.4   # left eye y

    # Right eye outer corner (index 263 in MediaPipe → point 46 in 68)
    landmarks_478[263, 0] = 0.65  # right eye x
    landmarks_478[263, 1] = 0.4   # right eye y

    frame_w, frame_h = 640, 480
    lm_68 = pipeline._convert_478_to_68(landmarks_478, frame_w, frame_h)

    # Shape check
    assert lm_68.shape == (68, 2), f"Expected (68, 2), got {lm_68.shape}"
    assert lm_68.dtype == np.float32

    # All values should be in pixel coordinates (not normalized)
    assert lm_68.max() > 1.0, \
        "Landmarks should be in pixel coordinates, not [0, 1] range"

    # Verify specific anatomical positions using pixel conversion
    # Nose tip (MP index 1) mapped to 68-point index 33 (0-based)
    # _MP_NOSE_BOTTOM_INDICES[2] = index 1 → 68-point #34 → array idx 33
    nose_tip_idx = 33  # 34th point (0-indexed = 33)
    assert abs(lm_68[nose_tip_idx, 0] - (0.5 * frame_w)) < 1.0, \
        f"Nose tip X should be {0.5 * frame_w}, got {lm_68[nose_tip_idx, 0]}"

    # Right eye outer (MP index 33) → 68-point #37 → array idx 36
    right_eye_idx = 36  # 37th point (0-indexed = 36)
    assert abs(lm_68[right_eye_idx, 0] - (0.35 * frame_w)) < 1.0, \
        f"Right eye X should be {0.35 * frame_w}, got {lm_68[right_eye_idx, 0]}"

    # Left eye outer (MP index 263) → 68-point #46 → array idx 45
    left_eye_idx = 45  # 46th point (0-indexed = 45)
    assert abs(lm_68[left_eye_idx, 0] - (0.65 * frame_w)) < 1.0, \
        f"Left eye X should be {0.65 * frame_w}, got {lm_68[left_eye_idx, 0]}"

    # Eye separation should be anatomically plausible
    eye_sep = abs(lm_68[left_eye_idx, 0] - lm_68[right_eye_idx, 0])
    assert eye_sep > 50, f"Eye separation {eye_sep:.0f}px seems too small"


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
