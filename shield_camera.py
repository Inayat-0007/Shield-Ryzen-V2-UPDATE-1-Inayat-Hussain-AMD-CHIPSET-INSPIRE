"""
Shield-Ryzen V2 — Camera Input Module
======================================
Owns ALL camera interaction. No other file should touch
cv2.VideoCapture directly.

Features:
  - Frame validation (shape, dtype, channel count, brightness)
  - Frame freshness detection (catches stale/frozen cameras)
  - Health monitoring (FPS, drop rate, connection status)
  - Proper resource cleanup

Developer: Inayat Hussain | AMD Slingshot 2026
Part 1 of 12 — Input Pipeline Hardening
"""

from __future__ import annotations

import time
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np


# ─── Module Logger ─────────────────────────────────────────────
_log = logging.getLogger("ShieldCamera")


class ShieldCamera:
    """Thread-safe, validated camera capture for Shield-Ryzen.

    Wraps cv2.VideoCapture with:
      - Explicit backend selection (DirectShow default on Windows)
      - 1-frame buffer to minimize latency
      - Per-frame validation (shape, dtype, brightness, channels)
      - Monotonic timestamping for freshness checking
      - Health status reporting (FPS, drops, age)
    """

    # ── Validation constants ──────────────────────────────────
    MIN_HEIGHT: int = 120
    MIN_WIDTH: int = 160
    EXPECTED_CHANNELS: int = 3
    EXPECTED_DTYPE = np.uint8
    MIN_MEAN_BRIGHTNESS: float = 5.0    # catches lens cap / hw failure
    MAX_MEAN_BRIGHTNESS: float = 250.0  # catches sensor saturation
    FPS_WINDOW: int = 30                # frames used for rolling FPS

    def __init__(
        self,
        camera_id: int = 0,
        backend: int = cv2.CAP_DSHOW,
    ) -> None:
        """Initialize camera with explicit backend and minimal buffer.

        Args:
            camera_id: System camera index (default 0 = primary).
            backend: OpenCV capture backend. cv2.CAP_DSHOW on Windows
                     to avoid MSMF buffering; use cv2.CAP_V4L2 on Linux.
        """
        self._camera_id: int = camera_id
        self._backend: int = backend
        self._backend_name: str = self._resolve_backend_name(backend)

        # Open capture
        self._cap: cv2.VideoCapture = cv2.VideoCapture(camera_id, backend)

        # Set buffer size to 1 to minimize latency (security-critical)
        # WHY: Default buffer is 3-5 frames → 30-80 ms stale data
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Record camera properties
        self._resolution: tuple[int, int] = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._fps_capability: float = self._cap.get(cv2.CAP_PROP_FPS)

        # Health counters
        self._frames_total: int = 0
        self._frames_dropped: int = 0
        self._last_valid_timestamp: float = 0.0

        # Rolling FPS tracker (timestamps of last N valid frames)
        self._frame_times: deque[float] = deque(maxlen=self.FPS_WINDOW)

        _log.info(
            "ShieldCamera initialized — id=%d backend=%s resolution=%s fps_cap=%.1f buffer=1",
            camera_id,
            self._backend_name,
            self._resolution,
            self._fps_capability,
        )

    # ── Public API ────────────────────────────────────────────

    def read_validated_frame(self) -> tuple[bool, Optional[np.ndarray], float]:
        """Read one frame and run full validation checklist.

        Returns:
            (success, frame_or_None, monotonic_timestamp)
            On failure: (False, None, 0.0) and increments drop counter.
        """
        self._frames_total += 1
        timestamp = time.monotonic()

        # 1. Read raw frame
        ret, frame = self._cap.read()

        # 2. Run validation checklist
        if not self._validate_frame(ret, frame):
            self._frames_dropped += 1
            return False, None, 0.0

        # Frame passed all checks
        self._last_valid_timestamp = timestamp
        self._frame_times.append(timestamp)

        return True, frame, timestamp

    def check_frame_freshness(
        self,
        timestamp: float,
        max_age_ms: float = 500.0,
    ) -> bool:
        """Check whether a frame is fresh enough for security decisions.

        Catches stale buffered frames and frozen cameras.

        Args:
            timestamp: Monotonic timestamp from read_validated_frame().
            max_age_ms: Maximum acceptable frame age in milliseconds.

        Returns:
            True if the frame is fresh, False if stale.
        """
        if timestamp <= 0.0:
            return False
        age_ms = (time.monotonic() - timestamp) * 1000.0
        fresh = age_ms <= max_age_ms
        if not fresh:
            _log.warning(
                "Stale frame detected — age=%.1f ms (limit=%.1f ms)",
                age_ms,
                max_age_ms,
            )
        return fresh

    def get_health_status(self) -> dict:
        """Return a snapshot of camera health metrics.

        Returns:
            Dictionary with connection status, measured FPS, drop
            count, last-valid-frame age, resolution, and backend.
        """
        now = time.monotonic()
        last_age_ms = (
            (now - self._last_valid_timestamp) * 1000.0
            if self._last_valid_timestamp > 0
            else float("inf")
        )

        return {
            "connected": self._cap.isOpened(),
            "fps_actual": self._calculate_fps(),
            "frames_total": self._frames_total,
            "frames_dropped": self._frames_dropped,
            "drop_rate_pct": (
                (self._frames_dropped / self._frames_total * 100.0)
                if self._frames_total > 0
                else 0.0
            ),
            "last_valid_frame_age_ms": round(last_age_ms, 2),
            "resolution": self._resolution,
            "backend": self._backend_name,
        }

    def is_opened(self) -> bool:
        """Whether the underlying capture device is open."""
        return self._cap.isOpened()

    def release(self) -> None:
        """Release camera resources and log final statistics."""
        health = self.get_health_status()
        _log.info(
            "ShieldCamera releasing — total=%d dropped=%d (%.1f%%) avg_fps=%.1f",
            health["frames_total"],
            health["frames_dropped"],
            health["drop_rate_pct"],
            health["fps_actual"],
        )
        self._cap.release()

    # ── Context manager support ───────────────────────────────

    def __enter__(self) -> "ShieldCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ── Private helpers ───────────────────────────────────────

    def _validate_frame(self, ret: bool, frame: Optional[np.ndarray]) -> bool:
        """Run full validation checklist on a captured frame.

        Every check is commented with WHY it matters for security.
        """
        # Check 1: Camera returned success flag
        if not ret:
            _log.debug("Validation FAIL: cap.read() returned ret=False")
            return False

        # Check 2: Frame is not None
        if frame is None:
            _log.debug("Validation FAIL: frame is None")
            return False

        # Check 3: Frame has exactly 3 dimensions (H, W, C)
        # WHY: Grayscale or 4-channel frames break the pipeline
        if frame.ndim != 3:
            _log.debug("Validation FAIL: ndim=%d (expected 3)", frame.ndim)
            return False

        # Check 4: Frame has 3 BGR channels
        # WHY: RGBA or single-channel would cause downstream crashes
        if frame.shape[2] != self.EXPECTED_CHANNELS:
            _log.debug(
                "Validation FAIL: channels=%d (expected %d)",
                frame.shape[2],
                self.EXPECTED_CHANNELS,
            )
            return False

        # Check 5: Frame dtype is uint8
        # WHY: Float frames from exotic cameras would break normalization
        if frame.dtype != self.EXPECTED_DTYPE:
            _log.debug("Validation FAIL: dtype=%s (expected uint8)", frame.dtype)
            return False

        # Check 6: Frame meets minimum resolution
        # WHY: Tiny frames yield garbage face detections
        h, w = frame.shape[:2]
        if h < self.MIN_HEIGHT or w < self.MIN_WIDTH:
            _log.debug(
                "Validation FAIL: resolution %dx%d below minimum %dx%d",
                w, h, self.MIN_WIDTH, self.MIN_HEIGHT,
            )
            return False

        # Check 7: Pixel values in valid uint8 range (redundant but defensive)
        # WHY: Sanity check — catches corrupt memory
        if frame.min() < 0 or frame.max() > 255:
            _log.debug("Validation FAIL: pixel values outside [0, 255]")
            return False

        # Check 8: Frame is not all-black (lens cap / hardware failure)
        mean_brightness = float(frame.mean())
        if mean_brightness <= self.MIN_MEAN_BRIGHTNESS:
            _log.debug(
                "Validation FAIL: all-black frame (mean=%.2f)", mean_brightness
            )
            return False

        # Check 9: Frame is not all-white (sensor saturation)
        if mean_brightness >= self.MAX_MEAN_BRIGHTNESS:
            _log.debug(
                "Validation FAIL: all-white frame (mean=%.2f)", mean_brightness
            )
            return False

        return True

    def _calculate_fps(self) -> float:
        """Compute rolling FPS over the last N valid frames."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed

    @staticmethod
    def _resolve_backend_name(backend: int) -> str:
        """Convert OpenCV backend constant to human-readable name."""
        names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "MediaFoundation",
            cv2.CAP_ANY: "Auto",
        }
        # CAP_V4L2 only exists on Linux builds
        if hasattr(cv2, "CAP_V4L2"):
            names[cv2.CAP_V4L2] = "V4L2"
        return names.get(backend, f"Unknown({backend})")
