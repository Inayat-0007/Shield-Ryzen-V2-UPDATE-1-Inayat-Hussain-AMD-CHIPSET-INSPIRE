"""
Shield-Ryzen V2 — Camera Module Tests
=======================================
8 test cases using synthetic NumPy frames — NO real camera needed.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 1 of 12 — Input Pipeline Hardening
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shield_camera import ShieldCamera


# ─── Fixtures ─────────────────────────────────────────────────

def _make_valid_frame(
    height: int = 480,
    width: int = 640,
    brightness: int = 128,
) -> np.ndarray:
    """Create a synthetic BGR frame that passes all validation checks."""
    rng = np.random.RandomState(42)
    frame = rng.randint(
        max(20, brightness - 60),
        min(240, brightness + 60),
        size=(height, width, 3),
        dtype=np.uint8,
    )
    return frame


def _make_mock_camera(frame: np.ndarray | None, ret: bool = True):
    """Create a mock cv2.VideoCapture that returns the given frame."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (ret, frame)
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0  # Default FPS
    mock_cap.set.return_value = True
    return mock_cap


# ─── Test 1: Valid frame passes all checks ────────────────────

def test_valid_frame_passes_all_checks():
    """A normal 480×640 BGR uint8 frame with typical brightness
    should pass all validation checks."""
    frame = _make_valid_frame()
    mock_cap = _make_mock_camera(frame, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, result_frame, ts = cam.read_validated_frame()

    assert ok is True, "Valid frame should return ok=True"
    assert result_frame is not None, "Valid frame should not be None"
    assert ts > 0, "Timestamp should be positive"
    assert np.array_equal(result_frame, frame), "Frame should be unchanged"
    cam.release()


# ─── Test 2: None frame returns False ─────────────────────────

def test_none_frame_returns_false():
    """When cap.read() returns None frame, validation should fail."""
    mock_cap = _make_mock_camera(frame=None, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, result_frame, ts = cam.read_validated_frame()

    assert ok is False, "None frame should fail validation"
    assert result_frame is None
    assert ts == 0.0
    cam.release()


# ─── Test 3: Wrong channels returns False ─────────────────────

def test_wrong_channels_returns_false():
    """A 4-channel BGRA frame should fail the channel check."""
    # 4 channels (BGRA) — not valid for our pipeline
    frame_4ch = np.full((480, 640, 4), 128, dtype=np.uint8)
    mock_cap = _make_mock_camera(frame_4ch, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, result_frame, ts = cam.read_validated_frame()

    assert ok is False, "4-channel frame should fail validation"
    assert result_frame is None
    cam.release()


# ─── Test 4: All-black frame returns False ────────────────────

def test_all_black_frame_returns_false():
    """An all-black frame (mean brightness ≤ 5) catches lens cap
    or hardware failure."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = _make_mock_camera(black_frame, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, result_frame, ts = cam.read_validated_frame()

    assert ok is False, "All-black frame should fail validation"
    assert result_frame is None
    cam.release()


# ─── Test 5: All-white frame returns False ────────────────────

def test_all_white_frame_returns_false():
    """An all-white frame (mean brightness ≥ 250) catches sensor
    saturation."""
    white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    mock_cap = _make_mock_camera(white_frame, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, result_frame, ts = cam.read_validated_frame()

    assert ok is False, "All-white frame should fail validation"
    assert result_frame is None
    cam.release()


# ─── Test 6: Undersized frame returns False ───────────────────

def test_undersized_frame_returns_false():
    """A frame below minimum resolution (120×160) should fail.
    100×100 is too small for face detection."""
    tiny_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    mock_cap = _make_mock_camera(tiny_frame, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, result_frame, ts = cam.read_validated_frame()

    assert ok is False, "Undersized frame should fail validation"
    assert result_frame is None
    cam.release()


# ─── Test 7: Stale timestamp detected ────────────────────────

def test_stale_timestamp_detected():
    """A frame older than max_age_ms should be flagged as stale.
    This catches frozen cameras and buffered frames."""
    frame = _make_valid_frame()
    mock_cap = _make_mock_camera(frame, ret=True)

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)
        ok, _, ts = cam.read_validated_frame()
        assert ok is True

        # Simulate a very old timestamp (1 second ago)
        old_timestamp = time.monotonic() - 1.0
        stale = cam.check_frame_freshness(old_timestamp, max_age_ms=500)
        assert stale is False, "1-second-old frame should be stale (limit 500ms)"

        # Fresh timestamp should be fine
        fresh = cam.check_frame_freshness(ts, max_age_ms=5000)
        assert fresh is True, "Just-captured frame should be fresh"

    cam.release()


# ─── Test 8: Health status reports correctly ──────────────────

def test_health_status_reports_correctly():
    """Health status dict should contain all required fields and
    reflect actual capture history."""
    valid_frame = _make_valid_frame()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    call_count = [0]
    frames = [valid_frame, black_frame, valid_frame]

    def mock_read():
        idx = min(call_count[0], len(frames) - 1)
        call_count[0] += 1
        return (True, frames[idx])

    mock_cap = MagicMock()
    mock_cap.read = mock_read
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cap.set.return_value = True

    with patch("shield_camera.cv2.VideoCapture", return_value=mock_cap):
        cam = ShieldCamera(camera_id=0)

        # Read 3 frames: valid, black (dropped), valid
        cam.read_validated_frame()
        cam.read_validated_frame()
        cam.read_validated_frame()

        health = cam.get_health_status()

    # Check all required fields exist
    assert "connected" in health
    assert "fps_actual" in health
    assert "frames_total" in health
    assert "frames_dropped" in health
    assert "drop_rate_pct" in health
    assert "last_valid_frame_age_ms" in health
    assert "resolution" in health
    assert "backend" in health

    # Check values
    assert health["connected"] is True
    assert health["frames_total"] == 3
    assert health["frames_dropped"] == 1, "Black frame should be counted as dropped"
    assert isinstance(health["fps_actual"], float)
    assert isinstance(health["resolution"], tuple)

    cam.release()


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
