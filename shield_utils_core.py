"""
Shield-Ryzen V2 — Shared Utility Module (Part 3 Rewrite)
==========================================================
Centralized security logic for all Shield-Ryzen engines.

Contains 5 Components:
  A) EAR with angle compensation (cosine correction at >15° yaw)
  B) Adaptive Laplacian analysis + FFT frequency forensics
  C) Device baseline calibration (Domain Adaptation)
  D) Confidence calibration via temperature scaling
  E) Temporal smoothing state machine (DecisionStateMachine)

Also:
  - Backward-compatible exports for v3_int8_engine.py
  - BlinkTracker with dynamic baseline scaling
  - SignalSmoother for temporal averaging
  - estimate_distance for proximity detection

═══════════════════════════════════════════════════════════
LOGIC COLLAPSE FIXES (Part 3):
  1. EAR now applies cos(yaw) correction for non-frontal angles
  2. Laplacian threshold is adaptive (40% of device baseline)
  3. FFT high-frequency ratio catches screens & GANs
  4. Temperature scaling prevents overconfident Softmax outputs
  5. Hysteresis prevents decision flickering
  6. Device calibration measures user's EAR range + camera noise
═══════════════════════════════════════════════════════════

Developer: Inayat Hussain | AMD Slingshot 2026
Part 3 of 14 — Liveness, Forensics & Decision Logic Calibration
"""

from __future__ import annotations

import cv2
import numpy as np
import math
import yaml
import os
import json
import time
import logging
from collections import deque
from typing import Optional


# ===================================================================
# Configuration
# ===================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_SCRIPT_DIR, 'config.yaml')


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from config.yaml."""
    target = path or _config_path
    with open(target, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


CONFIG = load_config()


# ===================================================================
# Constants (loaded from config.yaml, overridable at runtime)
# ===================================================================

CONFIDENCE_THRESHOLD = CONFIG['security']['confidence_threshold']
BLINK_THRESHOLD      = CONFIG['security']['blink_threshold']
BLINK_TIME_WINDOW    = CONFIG['security']['blink_time_window']
LAPLACIAN_THRESHOLD  = CONFIG['security']['laplacian_threshold']

MEAN       = np.array(CONFIG['preprocessing']['mean'], dtype=np.float32)
STD        = np.array(CONFIG['preprocessing']['std'],  dtype=np.float32)
INPUT_SIZE = CONFIG['preprocessing']['input_size']

LEFT_EYE   = CONFIG['landmarks']['left_eye']
RIGHT_EYE  = CONFIG['landmarks']['right_eye']

# Load decision thresholds (created by threshold_optimization)
_THRESHOLD_CONFIG_PATH = os.path.join(_SCRIPT_DIR, 'config', 'decision_thresholds.yaml')
_CALIBRATION_PATH = os.path.join(_SCRIPT_DIR, 'shield_calibration.json')

_decision_thresholds: dict = {}
if os.path.exists(_THRESHOLD_CONFIG_PATH):
    with open(_THRESHOLD_CONFIG_PATH, 'r', encoding='utf-8') as _f:
        _decision_thresholds = yaml.safe_load(_f) or {}

_device_calibration: dict = {}
if os.path.exists(_CALIBRATION_PATH):
    with open(_CALIBRATION_PATH, 'r', encoding='utf-8') as _f:
        _device_calibration = json.load(_f)


# ===================================================================
# Logging Setup
# ===================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger for Shield-Ryzen modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)-12s %(levelname)-7s %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


_log = setup_logger('ShieldUtils')


# ===================================================================
# Face Preprocessing (backward compatible — FF++ standard)
# ===================================================================

def preprocess_face(face_crop: np.ndarray) -> np.ndarray:
    """Preprocess face crop for Xception: resize 299x299, normalize [-1,1], NCHW.

    NORMALIZATION: FF++ standard — (pixel/255 - 0.5) / 0.5
    This maps [0, 255] -> [-1.0, +1.0]
    """
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    face_float = face_resized.astype(np.float32) / 255.0
    face_norm = (face_float - MEAN) / STD
    face_chw = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0).astype(np.float32)


# ===================================================================
# COMPONENT A: EAR WITH ANGLE COMPENSATION
# ===================================================================

def compute_ear(
    landmarks,
    eye_indices: list[int],
    head_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    is_frontal: bool = True,
) -> tuple[float, str]:
    """Compute Eye Aspect Ratio with COSINE ANGLE COMPENSATION.

    LOGIC COLLAPSE FIX:
      1. Uses PIXEL coordinates (not normalized 0-1!)
         Verify: p1 = landmarks[33]  →  (x_px, y_px)
         NOT:    p1 = (landmarks[33].x, landmarks[33].y)  →  normalized!
      2. Standard EAR formula for frontal faces
      3. If |yaw| > 15°: corrected_ear = raw_ear / cos(yaw_radians)
         WHY: At angle, horizontal eye distance shortens in 2D projection,
         artificially inflating EAR. Cosine corrects for this foreshortening.
      4. If |yaw| > 25°: return (ear, "LOW") — unreliable

    Args:
        landmarks: Array or list of (x, y) pixel landmarks. Supports both
                   object (.x, .y) and array/tuple ([0], [1]) formats.
        eye_indices: 6 indices [outer, upper1, upper2, inner, lower2, lower1]
        head_pose: (yaw, pitch, roll) in degrees
        is_frontal: True if face is frontal (from pipeline)

    Returns:
        (ear_value, reliability_grade)
        reliability_grade: "HIGH" if frontal (|yaw|<15, |pitch|<25)
                          "MEDIUM" if 15-25° off
                          "LOW" if >25° off or invalid
    """
    try:
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            # Support both object (.x, .y) and array/tuple ([0], [1]) formats
            if hasattr(lm, 'x') and hasattr(lm, 'y'):
                points.append((float(lm.x), float(lm.y)))
            else:
                points.append((float(lm[0]), float(lm[1])))

        p1, p2, p3, p4, p5, p6 = points

        # Vertical distances (Eyelid opening)
        v1 = math.sqrt((p2[0] - p6[0])**2 + (p2[1] - p6[1])**2)
        v2 = math.sqrt((p3[0] - p5[0])**2 + (p3[1] - p5[1])**2)

        # Horizontal distance (Eye width)
        h_dist = math.sqrt((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2)

        if h_dist < 1e-6:
            return 0.05, "LOW"

        ear = (v1 + v2) / (2.0 * h_dist)

        # HEAD POSE VALIDATION & COSINE CORRECTION
        yaw, pitch, roll = head_pose
        reliability = "HIGH"

        # Pitch validation (relaxed for natural webcam use)
        if pitch > 30.0:
            # Looking down significantly — eyelids naturally lower
            reliability = "LOW"
        elif pitch < -25.0:
            # Looking up significantly
            reliability = "MEDIUM"

        # Yaw validation + cosine correction
        # Relaxed from 15/25 to 20/30 because most webcam users sit
        # slightly off-center, causing unnecessary MEDIUM reliability
        abs_yaw = abs(yaw)
        if abs_yaw > 30.0:
            # Extreme angle — EAR unreliable regardless of correction
            reliability = "LOW"
            # Still apply correction for best-effort value
            cos_yaw = max(0.3, math.cos(math.radians(abs_yaw)))
            ear = ear / cos_yaw
        elif abs_yaw > 20.0:
            # Moderate angle — apply cosine correction, mark MEDIUM
            if reliability == "HIGH":
                reliability = "MEDIUM"
            cos_yaw = math.cos(math.radians(abs_yaw))
            ear = ear / cos_yaw

        return round(float(ear), 4), reliability

    except (IndexError, TypeError, ValueError):
        return 0.05, "LOW"


# ===================================================================
# COMPONENT B: ADAPTIVE LAPLACIAN + FREQUENCY ANALYSIS
# ===================================================================

def compute_texture_score(
    face_crop_raw: np.ndarray,
    device_baseline: Optional[float] = None,
    distance_cm: float = 0.0,
) -> tuple[float, bool, str]:
    """Enhanced texture analysis with 5-LAYER SCREEN REPLAY DETECTION.

    Layer 1: Laplacian Variance (standard sharpness)
    Layer 2: Physics-based distance-texture cross-validation
    Layer 3: Moiré pattern detection (screen pixel grid interference)
    Layer 4: Screen light emission detection (catches CLOSE-RANGE attacks)
    Layer 5: Combined weak-signal fusion (2+ marginal signals = confirm)

    Returns:
        (laplacian_variance, is_suspicious, explanation)
    """
    if face_crop_raw is None or face_crop_raw.size == 0:
        return 0.0, True, "EMPTY_CROP"

    gray = cv2.cvtColor(face_crop_raw, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Forehead ROI for stability (15-35% rows, 25-75% cols)
    forehead = gray[int(h * 0.15):int(h * 0.35), int(w * 0.25):int(w * 0.75)]
    if forehead.size == 0:
        forehead = gray

    # ── LAYER 1: Laplacian Variance ──
    lap_var = float(cv2.Laplacian(forehead, cv2.CV_64F).var())

    if device_baseline and device_baseline > 0:
        thresh_low = device_baseline * 0.4
    else:
        thresh_low = 15.0

    # ── LAYER 2: Distance-Texture Physics Check ──
    screen_replay_flag = False
    screen_confidence = 0.0
    screen_signals = []  # Track signals for Layer 5 fusion

    REFERENCE_TEXTURE = 700.0   # Real faces produce 400-950 at 50cm (verified from screenshots)
    REFERENCE_DISTANCE = 50.0

    if distance_cm > 40:
        max_expected = REFERENCE_TEXTURE * (REFERENCE_DISTANCE / distance_cm) ** 2
        max_allowed = max_expected * 4.0  # 4x safety margin — real face variance is extreme

        if lap_var > max_allowed:
            screen_replay_flag = True
            sig = min((lap_var / max_allowed - 1.0) / 2.0, 1.0)
            screen_confidence = max(screen_confidence, sig)
            screen_signals.append(("physics", sig))

    ABSOLUTE_TEXTURE_CAP = 1500.0  # Real faces reach 950 at 46cm; screens produce 2000+
    if lap_var > ABSOLUTE_TEXTURE_CAP and not screen_replay_flag:
        screen_replay_flag = True
        sig = min((lap_var - ABSOLUTE_TEXTURE_CAP) / 500.0, 0.9)
        screen_confidence = max(screen_confidence, sig)
        screen_signals.append(("abs_cap", sig))

    # ── LAYER 3: Frequency Analysis + Moiré Detection ──
    hf_ratio = _compute_hf_energy_ratio(forehead)
    moire_score = _detect_moire_pattern(forehead)

    # Raised to 0.50: real skin noise scores 0.25-0.40 and was false-positive
    if moire_score > 0.50:
        sig = min(moire_score * 1.2, 1.0)
        screen_signals.append(("moire", sig))
        if moire_score > 0.70:  # Very strong = definite screen
            screen_replay_flag = True
            screen_confidence = max(screen_confidence, sig)

    # ── LAYER 4: Screen Light Emission Detection ──
    # Screens EMIT light (backlit LCD/OLED), creating 3 anomalies:
    #   A) Abnormally UNIFORM brightness across the face
    #   B) Narrow skin chrominance range (synthetic colors)
    #   C) Blue/green color cast from LED backlighting
    screen_light_score = _detect_screen_light(face_crop_raw)
    if screen_light_score > 0.50:  # Raised: indoor LED lighting false-positives at 0.30
        screen_signals.append(("screen_light", screen_light_score))
        if screen_light_score > 0.70:  # Only confirm on very strong signal
            screen_replay_flag = True
            screen_confidence = max(screen_confidence, screen_light_score)

    # ── LAYER 5: Combined Weak-Signal Fusion ──
    # CONSERVATIVE: Require 3+ signals above 0.35 to prevent false positives.
    # Indoor lighting can produce 1-2 weak signals on real faces.
    active_signals = [(n, s) for n, s in screen_signals if s > 0.35]
    if len(active_signals) >= 3 and not screen_replay_flag:
        combined = sum(s for _, s in active_signals) / len(active_signals)
        if combined > 0.45:
            screen_replay_flag = True
            screen_confidence = max(screen_confidence, combined)
            screen_signals.append(("fusion", combined))

    # ── COMBINED SUSPICION LOGIC ──
    suspicious = False
    reasons = []

    if lap_var < thresh_low:
        suspicious = True
        reasons.append(f"Laplacian {lap_var:.1f} < threshold {thresh_low:.1f}")

    if hf_ratio < 0.08:
        suspicious = True
        reasons.append(f"HF ratio {hf_ratio:.3f} < 0.08 (spectral anomaly)")

    if screen_replay_flag:
        suspicious = True
        sig_str = "+".join(f"{n}={s:.0%}" for n, s in screen_signals)
        reasons.append(f"SCREEN REPLAY (conf={screen_confidence:.0%}, signals=[{sig_str}])")

    # Build explanation string — always include raw signal values for debugging
    baseline_str = f"adaptive baseline={device_baseline}" if device_baseline else "default"
    explain = f"LAP: {lap_var:.1f}/{thresh_low:.1f} | HF: {hf_ratio:.3f} | moire={moire_score:.3f} | light={screen_light_score:.3f} | {baseline_str}"
    if screen_replay_flag:
        explain += f" | ⚠ SCREEN_REPLAY (conf={screen_confidence:.0%}, moire={moire_score:.2f}, light={screen_light_score:.2f})"
    if suspicious:
        explain += f" — FLAGGED: {'; '.join(reasons)}"

    # DEBUG: Write raw signal values to file for analysis
    try:
        import time as _t
        _dbg_line = (f"{_t.time():.1f} | LAP={lap_var:.1f} DIST={distance_cm:.0f}cm "
                     f"MOIRE={moire_score:.3f} LIGHT={screen_light_score:.3f} "
                     f"HF={hf_ratio:.3f} REPLAY={screen_replay_flag} "
                     f"SUSP={suspicious}\n")
        _dbg_path = os.path.join(os.path.dirname(__file__), "logs", "texture_debug.log")
        with open(_dbg_path, "a") as _df:
            _df.write(_dbg_line)
    except Exception:
        pass

    return lap_var, suspicious, explain


def _compute_hf_energy_ratio(gray_roi: np.ndarray) -> float:
    """Compute high-frequency energy ratio via FFT for forensic analysis."""
    if gray_roi.size < 16:
        return 0.5

    f = np.fft.fft2(gray_roi.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1e-10)

    cy, cx = gray_roi.shape[0] // 2, gray_roi.shape[1] // 2
    r_hf = int(min(gray_roi.shape) * 0.25)

    y, x = np.ogrid[:gray_roi.shape[0], :gray_roi.shape[1]]
    mask_lf = (y - cy)**2 + (x - cx)**2 <= r_hf**2

    hf_energy = mag[~mask_lf].mean() if not np.all(mask_lf) else 0
    lf_energy = mag[mask_lf].mean() + 1e-10

    return hf_energy / lf_energy


def _detect_moire_pattern(gray_roi: np.ndarray) -> float:
    """Detect Moiré interference patterns from screen pixel grids.

    Screens have regular pixel grids that create periodic frequency
    peaks when viewed through a camera lens. Real skin has NO periodic
    spatial patterns.

    Method:
      1. Compute 2D FFT magnitude spectrum
      2. Build radial profile (energy vs spatial frequency)
      3. Look for periodic spikes in the radial profile
      4. Score the periodicity using autocorrelation of the radial profile

    Returns:
        Moiré score in [0.0, 1.0]. >0.6 = likely screen replay.
    """
    if gray_roi.size < 64 or min(gray_roi.shape) < 16:
        return 0.0

    try:
        # FFT
        f = np.fft.fft2(gray_roi.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)

        h, w = gray_roi.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        # Build radial profile
        radial_bins = max_r
        profile = np.zeros(radial_bins)
        counts = np.zeros(radial_bins)

        y_coords, x_coords = np.mgrid[:h, :w]
        r_map = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2).astype(int)

        for r in range(radial_bins):
            mask = r_map == r
            if np.any(mask):
                profile[r] = mag[mask].mean()
                counts[r] = np.sum(mask)

        # Normalize and analyze profile for periodic peaks
        # Remove DC component and very low frequencies
        if radial_bins < 10:
            return 0.0

        profile_detrended = profile[3:]  # Skip DC + low-freq
        if len(profile_detrended) < 8:
            return 0.0

        # Normalize
        p_mean = profile_detrended.mean()
        if p_mean < 1e-6:
            return 0.0
        p_norm = profile_detrended / p_mean

        # Detect peaks: values > 2× mean indicate periodic structure
        peaks = p_norm > 2.0
        n_peaks = np.sum(peaks)
        peak_ratio = n_peaks / len(p_norm)

        # Autocorrelation of radial profile (periodic signals have high autocorrelation)
        profile_centered = p_norm - p_norm.mean()
        autocorr = np.correlate(profile_centered, profile_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]  # Normalize

        # Find periodicity: look for secondary peaks in autocorrelation
        secondary_peaks = 0
        for i in range(3, min(len(autocorr) - 1, 30)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.15:
                secondary_peaks += 1

        # Score: combine peak count and autocorrelation periodicity
        moiré_score = 0.0
        if peak_ratio > 0.1:   # Many spectral peaks
            moiré_score += 0.3
        if peak_ratio > 0.2:
            moiré_score += 0.2
        if secondary_peaks >= 2:  # Periodic structure in autocorrelation
            moiré_score += 0.3
        if secondary_peaks >= 4:
            moiré_score += 0.2

        return min(moiré_score, 1.0)

    except Exception:
        return 0.0


def _detect_screen_light(face_crop_bgr: np.ndarray) -> float:
    """Detect screen light emission characteristics.

    Screens emit their own light (backlit LCD/OLED), creating 3 physical
    anomalies that real faces lit by ambient light don't have:

    Signal A: BRIGHTNESS UNIFORMITY
      - Screen backlight is spatially uniform → face brightness varies little
      - Real faces: highlights on nose/forehead, shadows under chin/eyes
      - Metric: coefficient of variation (std/mean) of face luminance
      - Real face CV: 0.25-0.55 (high variance from 3D face geometry)
      - Screen face CV: 0.08-0.18 (flat, uniform backlight)

    Signal B: CHROMINANCE RANGE
      - Screen RGB subpixels produce a narrower color gamut than reflected light
      - Real skin has rich chrominance variation (blood flow, melanin patches)
      - Metric: standard deviation of Cr and Cb channels in YCrCb space
      - Real face: Cr_std > 8, Cb_std > 6
      - Screen face: Cr_std < 6, Cb_std < 5

    Signal C: BLUE CHANNEL BIAS
      - LED backlights (especially white LEDs) emit more blue light
      - This creates a measurable blue shift in the face region
      - Metric: blue_channel_mean / green_channel_mean ratio
      - Real face: B/G ≈ 0.75-0.90 (warm skin tone)
      - Screen face: B/G ≈ 0.88-1.05 (blue-shifted backlight)

    Returns:
        Screen light score in [0.0, 1.0]. >0.55 = likely screen.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0

    try:
        h, w = face_crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return 0.0

        # Use center 60% of face crop (avoid border artifacts)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        center = face_crop_bgr[y1:y2, x1:x2]

        # ── Signal A: Brightness Uniformity ──
        gray_center = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_brightness = gray_center.mean()
        if mean_brightness < 10:  # Too dark to analyze
            return 0.0
        cv_brightness = gray_center.std() / mean_brightness  # coefficient of variation

        # Score: lower CV = more uniform = more likely screen
        # Real face: CV 0.25-0.55 → score 0.0
        # Screen:    CV 0.08-0.18 → score 0.3-0.5
        brightness_score = 0.0
        if cv_brightness < 0.22:
            brightness_score = max(0.0, (0.22 - cv_brightness) / 0.14)  # 0.22→0, 0.08→1.0
            brightness_score = min(brightness_score, 1.0)

        # ── Signal B: Chrominance Range ──
        ycrcb = cv2.cvtColor(center, cv2.COLOR_BGR2YCrCb)
        cr_std = float(ycrcb[:, :, 1].astype(np.float32).std())
        cb_std = float(ycrcb[:, :, 2].astype(np.float32).std())

        # Score: narrower chrominance = more likely screen
        chroma_score = 0.0
        if cr_std < 8.0:
            chroma_score += max(0.0, (8.0 - cr_std) / 5.0) * 0.5
        if cb_std < 6.0:
            chroma_score += max(0.0, (6.0 - cb_std) / 4.0) * 0.5
        chroma_score = min(chroma_score, 1.0)

        # ── Signal C: Blue Channel Bias ──
        b_mean = float(center[:, :, 0].astype(np.float32).mean())
        g_mean = float(center[:, :, 1].astype(np.float32).mean())
        if g_mean < 10:
            bg_ratio = 0.85  # neutral fallback
        else:
            bg_ratio = b_mean / g_mean

        # Score: B/G > 0.92 = blue-shifted (likely screen backlight)
        blue_score = 0.0
        if bg_ratio > 0.92:
            blue_score = min((bg_ratio - 0.92) / 0.15, 1.0)  # 0.92→0, 1.07→1.0

        # ── Combined screen light score ──
        # Weight: brightness uniformity most reliable, then chroma, then blue
        total = (brightness_score * 0.45 +
                 chroma_score * 0.35 +
                 blue_score * 0.20)

        return min(total, 1.0)

    except Exception:
        return 0.0


# ===================================================================
# COMPONENT C: DEVICE BASELINE CALIBRATION
# ===================================================================

def calibrate_device_baseline(
    face_crop_raw: np.ndarray = None,
    camera=None,
    face_pipeline=None,
    num_frames: int = 100,
    output_path: str = None,
) -> dict:
    """Domain Adaptation: Calibrate thresholds to THIS device.

    Run at first startup or user request. Captures frames and
    measures baseline sensor characteristics:
      - Laplacian variance distribution (camera sharpness)
      - User's natural EAR range (baseline for blink detection)
      - Lighting condition estimate
      - Camera resolution

    Saves results to shield_calibration.json for runtime use.

    Args:
        face_crop_raw: Optional single frame (backward compat).
        camera: ShieldCamera instance for multi-frame calibration.
        face_pipeline: ShieldFacePipeline for face detection during cal.
        num_frames: Number of frames to sample.
        output_path: Path to save calibration JSON.

    Returns:
        Calibration dictionary with recommended thresholds.
    """
    import datetime

    lap_values: list[float] = []
    ear_values: list[float] = []
    resolutions: list[tuple[int, int]] = []

    if face_crop_raw is not None:
        # Single-frame mode (backward compat)
        gray = cv2.cvtColor(face_crop_raw, cv2.COLOR_BGR2GRAY)
        lap_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lap_values.append(lap_val)
        num_frames = 1
    elif camera is not None:
        # Camera mode: capture frames
        for i in range(num_frames):
            if hasattr(camera, 'read_validated_frame'):
                ok, frame, ts = camera.read_validated_frame()
                if not ok or frame is None:
                    continue
            else:
                ret, frame = camera.read()
                if not ret or frame is None:
                    continue

            h, w = frame.shape[:2]
            resolutions.append((w, h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            # Detect face and measure EAR if pipeline available
            if face_pipeline is not None:
                try:
                    faces = face_pipeline.detect_faces(frame)
                    if faces:
                        face = faces[0]
                        lm = face.landmarks
                        if lm.shape[0] >= 400:
                            ear_l, _ = compute_ear(lm, LEFT_EYE, face.head_pose)
                            ear_r, _ = compute_ear(lm, RIGHT_EYE, face.head_pose)
                            ear_values.append((ear_l + ear_r) / 2.0)
                except Exception:
                    pass

            # ~30 FPS pacing
            time.sleep(0.033)
    else:
        # Synthetic mode: generate random frames for testing
        rng = np.random.RandomState(42)
        for _ in range(num_frames):
            synth = rng.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(synth, cv2.COLOR_BGR2GRAY)
            lap_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            ear_values.append(0.28 + rng.uniform(-0.02, 0.02))

    if not lap_values:
        lap_values = [30.0]

    lap_mean = float(np.mean(lap_values))
    lap_std = float(np.std(lap_values))
    ear_mean = float(np.mean(ear_values)) if ear_values else 0.28
    ear_std = float(np.std(ear_values)) if len(ear_values) > 1 else 0.03

    # Estimate lighting condition from average Laplacian variance
    if lap_mean > 100:
        lighting = "GOOD"
    elif lap_mean > 40:
        lighting = "MODERATE"
    else:
        lighting = "LOW_LIGHT"

    # Compute recommended thresholds
    recommended_lap_threshold = round(lap_mean * 0.4, 2)
    recommended_ear_threshold = round(max(0.15, ear_mean - 2.5 * ear_std), 4)

    # Camera resolution
    if resolutions:
        res = resolutions[0]
    elif camera is not None:
        res = ("live", "live")
    else:
        res = ("synthetic", "synthetic")

    result = {
        "laplacian_mean": round(lap_mean, 2),
        "laplacian_std": round(lap_std, 2),
        "recommended_threshold": recommended_lap_threshold,
        "ear_baseline_mean": round(ear_mean, 4),
        "ear_baseline_std": round(ear_std, 4),
        "recommended_ear_threshold": recommended_ear_threshold,
        "camera_resolution": f"{res[0]}x{res[1]}" if isinstance(res[0], int) else str(res[0]),
        "lighting_condition": lighting,
        "calibration_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "calibration_frames_used": len(lap_values),
        "num_frames_captured": len(lap_values),
    }

    if output_path:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
            exist_ok=True,
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        _log.info("Calibration saved: %s (lap_mean=%.1f, ear_mean=%.3f)",
                  output_path, lap_mean, ear_mean)

    return result


# ===================================================================
# COMPONENT D: CONFIDENCE CALIBRATION (Temperature Scaling)
# ===================================================================

class ConfidenceCalibrator:
    """Temperature scaling to fix overconfident Softmax outputs.

    WHY: Raw Softmax outputs 0.99 when it should output ~0.7.
    This causes the 89% threshold to be meaningless because
    almost everything passes it.

    HOW: Divides logits by temperature before re-applying Softmax.
      temperature > 1.0 = softer (less overconfident)
      temperature < 1.0 = sharper (more confident)
      temperature = 1.0 = identity (no change)
    Default 1.5 is conservative — reduces overconfidence without
    destroying discrimination.

    MATH:
      1. logits = log(softmax_probs + eps)
      2. scaled_logits = logits / temperature
      3. calibrated = softmax(scaled_logits)
    """

    def __init__(self, temperature: float = 1.5):
        """Initialize temperature scaling calibrator.

        Args:
            temperature: Scaling factor. >1 = softer, <1 = sharper.
        """
        self.temperature = temperature
        self.temp = temperature  # Alias for backward compat

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Convert softmax→logits, apply temperature, re-softmax.

        Args:
            probs: Raw softmax output array (e.g. [fake_prob, real_prob]).

        Returns:
            Temperature-scaled probability array summing to 1.0.
        """
        logits = np.log(np.clip(probs, 1e-10, 1.0))
        scaled = logits / self.temperature
        # Numerically stable softmax
        exp_scaled = np.exp(scaled - scaled.max())
        return exp_scaled / exp_scaled.sum()

    def save_params(self, path: str = None) -> None:
        """Save calibration parameters to JSON."""
        if path is None:
            path = os.path.join(_SCRIPT_DIR, 'config', 'temperature_params.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    @classmethod
    def load_params(cls, path: str = None) -> "ConfidenceCalibrator":
        """Load calibration parameters from JSON."""
        if path is None:
            path = os.path.join(_SCRIPT_DIR, 'config', 'temperature_params.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            return cls(temperature=params["temperature"])
        return cls()  # Default


# ===================================================================
# Signal Smoother (Temporal averaging for per-face state)
# ===================================================================

class SignalSmoother:
    """Exponential moving average for temporal signal smoothing.

    Used per-face-ID to smooth neural confidence and texture scores
    across frames, preventing single-frame spikes from causing
    state transitions.
    """

    def __init__(self, alpha: float = 0.3):
        """Initialize smoother.

        Args:
            alpha: Weight of new value. Higher = less smoothing.
                   0.3 = moderate smoothing (default for neural).
                   0.15 = heavy smoothing (texture scores).
        """
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, new_value: float) -> float:
        """Update with new observation, return smoothed value."""
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.alpha * new_value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        """Reset to uninitialized state."""
        self._value = None

    @property
    def value(self) -> float:
        """Current smoothed value."""
        return self._value if self._value is not None else 0.0


# ===================================================================
# COMPONENT E: TEMPORAL SMOOTHING STATE MACHINE
# ===================================================================

class DecisionStateMachine:
    """Security Truth Table Fusion with Temporal Hysteresis.

    States: VERIFIED, SUSPICIOUS, HIGH_RISK, UNKNOWN,
            REAL, WAIT_BLINK, FAKE, CRITICAL,
            NO_FACE, CAMERA_ERROR

    CONFLICT RESOLUTION TRUTH TABLE:

    Neural | Liveness | Forensic | → Decision
    -------|----------|----------|----------
    Real   | Pass     | Pass     | → REAL (engine promotes to VERIFIED)
    Real   | Pass     | Fail     | → SUSPICIOUS
    Real   | Fail     | Pass     | → WAIT_BLINK
    Real   | Fail     | Fail     | → HIGH_RISK
    Fake   | Pass     | Pass     | → FAKE
    Fake   | Pass     | Fail     | → FAKE
    Fake   | Fail     | Pass     | → FAKE
    Fake   | Fail     | Fail     | → CRITICAL

    AGILE HYSTERESIS:
    - Security escalation (→ FAKE/CRITICAL/HIGH_RISK): immediate (1 frame)
    - Safety de-escalation (→ REAL/VERIFIED): requires N frames

    This accepts BOTH string-based and TierResult inputs for
    backward compatibility with v3_int8_engine.py.
    """

    def __init__(self, frames: int = 5):
        """Initialize state machine.

        Args:
            frames: Hysteresis frames for de-escalation transitions.
                    Higher = more stable but slower to verify.
        """
        self.hysteresis = frames
        self.history: list[str] = []
        self.state = "UNKNOWN"
        self.counter = 0
        self._pending_state: Optional[str] = None
        self._state_entry_time = time.monotonic()
        self._total_transitions = 0
        self._rolling_history: deque = deque(maxlen=300)  # 10 sec at 30 FPS

    def update(self, t1, t2, t3, **kwargs) -> str:
        """Apply truth table fusion + hysteresis.

        Accepts both string-based (from engine) and TierResult (from
        calibrated_decision module) inputs.

        Args:
            t1: Neural tier — "REAL"/"FAKE" or TierResult(passed=bool)
            t2: Liveness tier — "PASS"/"FAIL" or TierResult
            t3: Forensic tier — "PASS"/"FAIL" or TierResult

        Returns:
            Stable state after hysteresis.
        """
        # Normalize inputs to booleans
        n = self._to_bool(t1, "REAL")
        l = self._to_bool(t2, "PASS")
        f = self._to_bool(t3, "PASS")

        # Truth table: 3-bit encoding
        case_id = (int(n) << 2) | (int(l) << 1) | int(f)

        table = {
            7: "REAL",        # [1,1,1] - All pass → REAL (engine upgrades to VERIFIED)
            6: "SUSPICIOUS",  # [1,1,0] - Texture/Forensic issue
            5: "WAIT_BLINK",  # [1,0,1] - Neural OK, Texture OK, but no Blink
            4: "HIGH_RISK",   # [1,0,0] - Only Neural passes
            3: "FAKE",        # [0,1,1] - Neural fail
            2: "FAKE",        # [0,1,0]
            1: "FAKE",        # [0,0,1]
            0: "CRITICAL",    # [0,0,0]
        }
        proposed = table.get(case_id, "UNKNOWN")

        # Record to rolling history
        self._rolling_history.append({
            "timestamp": time.monotonic(),
            "proposed": proposed,
            "stable": self.state,
            "tier_inputs": (n, l, f),
        })

        # First transition from UNKNOWN: apply immediately
        if self.state == "UNKNOWN":
            self.state = proposed
            self.counter = 0
            self._state_entry_time = time.monotonic()
            return self.state

        # AGILE HYSTERESIS:
        # Immediate escalation for security states (FAKE/CRITICAL/HIGH_RISK)
        # Slow de-escalation for safety states (REAL/VERIFIED)
        security_states = {"FAKE", "CRITICAL", "HIGH_RISK"}
        threshold = 1 if proposed in security_states else self.hysteresis

        if proposed == self.state:
            self.counter = 0
        else:
            if proposed == self._pending_state:
                self.counter += 1
            else:
                self._pending_state = proposed
                self.counter = 1

            if self.counter >= threshold:
                old_state = self.state
                self.state = proposed
                self.counter = 0
                self._pending_state = None
                self._total_transitions += 1
                self._state_entry_time = time.monotonic()

        return self.state

    @staticmethod
    def _to_bool(value, true_label: str) -> bool:
        """Convert string or TierResult to boolean."""
        if hasattr(value, 'passed'):
            return value.passed
        return str(value).upper() == true_label.upper()

    def get_state_duration_ms(self) -> float:
        """How long the current state has been active (ms)."""
        return (time.monotonic() - self._state_entry_time) * 1000.0

    def get_stability_score(self) -> float:
        """Score: 0.0 = flickering, 1.0 = stable."""
        if len(self._rolling_history) < 10:
            return 0.5
        recent = list(self._rolling_history)[-30:]
        proposed_states = [h["proposed"] for h in recent]
        unique_states = len(set(proposed_states))
        return round(max(0.0, 1.0 - (unique_states - 1) / 3.0), 3)

    def get_summary(self) -> dict:
        """Return current state machine summary."""
        return {
            "current_state": self.state,
            "pending_state": self._pending_state,
            "pending_count": self.counter,
            "hysteresis_threshold": self.hysteresis,
            "state_duration_ms": round(self.get_state_duration_ms(), 1),
            "stability_score": self.get_stability_score(),
            "total_transitions": self._total_transitions,
            "history_length": len(self._rolling_history),
        }

    def reset(self) -> None:
        """Reset to UNKNOWN."""
        self.state = "UNKNOWN"
        self._pending_state = None
        self.counter = 0
        self._rolling_history.clear()
        self._state_entry_time = time.monotonic()


# ===================================================================
# BlinkTracker (Dynamic Baseline Scaling)
# ===================================================================

class BlinkTracker:
    """Robust Blink Detection using Dynamic Baseline Scaling (DBS).

    Adapts to individual eye openness in real-time.
    Supports dual input: Blendshape-first (higher quality) with
    EAR-DBS fallback for cameras without blendshape support.

    BLINK TEMPORAL PATTERN (last 60 frames):
    - Natural blink: 100-400ms duration, 15-20 per minute
    - Suspicious: exactly periodic (robotic)
    - Suspicious: no blinking for >10 seconds
    - Return blink_pattern_score: 0.0=suspicious, 1.0=natural
    """

    def __init__(self, ear_threshold: float = 0.20):
        """Initialize blink tracker.

        Args:
            ear_threshold: ignored (dynamic baseline used instead)
        """
        self.open_state_ear = 0.30  # Initial assumption (adapts instantly)
        self.blink_count = 0
        self.history: deque = deque(maxlen=300)
        self.event_times: list[float] = []

        # State Machine
        self._in_blink = False
        self._blink_start_time = 0.0
        self._blink_peak_ear = 1.0

    def update(
        self,
        ear: float,
        timestamp: float,
        reliability: str,
        blendshapes: Optional[list] = None,
    ) -> dict:
        """Process one frame's EAR/blendshape data.

        Args:
            ear: Eye Aspect Ratio (already cosine-corrected by compute_ear)
            timestamp: time.monotonic() value
            reliability: "HIGH", "MEDIUM", or "LOW"
            blendshapes: Optional MediaPipe blendshape list

        Returns:
            dict with blink_detected, count, pattern_score, reliability, etc.
        """
        self.history.append((timestamp, ear))
        detected = False
        source = "EAR_DBS"

        # PRIORITY 1: Blendshape detection (higher quality)
        if blendshapes:
            try:
                # MediaPipe: 9=EyeBlinkLeft, 10=EyeBlinkRight
                bs_left = blendshapes[9].score
                bs_right = blendshapes[10].score
                avg_blink = (bs_left + bs_right) / 2.0
                source = "BLENDSHAPES"

                is_closed = avg_blink > 0.5

                if reliability in ("HIGH", "MEDIUM"):
                    if not self._in_blink and is_closed:
                        self._in_blink = True
                        self._blink_start_time = timestamp

                    elif self._in_blink and not is_closed:
                        self._in_blink = False
                        duration = timestamp - self._blink_start_time
                        # Real blinks: 100-400ms. Reject <60ms (noise) and >500ms (long close)
                        if 0.06 < duration < 0.50:
                            self.blink_count += 1
                            self.event_times.append(timestamp)
                            detected = True
                    elif self._in_blink:
                        # Safety: if stuck in blink for >1 second, auto-reset
                        if (timestamp - self._blink_start_time) > 1.0:
                            self._in_blink = False

                return {
                    "blink_detected": detected,
                    "count": self.blink_count,
                    "pattern_score": self._score_pattern(),
                    "reliability": reliability,
                    "baseline": 0.0,
                    "thresh": 0.5,
                    "source": source,
                }
            except (IndexError, AttributeError):
                pass  # Fallback to EAR

        # PRIORITY 2: EAR with Dynamic Baseline Scaling
        # Update baseline (ceiling of signal = open eye state)
        if ear > self.open_state_ear:
            self.open_state_ear = 0.9 * self.open_state_ear + 0.1 * ear
        else:
            self.open_state_ear = 0.998 * self.open_state_ear + 0.002 * ear

        # Relative thresholds
        thresh_close = self.open_state_ear * 0.65  # Close if <65% of open
        thresh_open = self.open_state_ear * 0.90   # Reopen if >90% of open

        if reliability in ("HIGH", "MEDIUM"):
            if not self._in_blink and ear < thresh_close:
                self._in_blink = True
                self._blink_start_time = timestamp
                self._blink_peak_ear = ear

            elif self._in_blink:
                if ear < self._blink_peak_ear:
                    self._blink_peak_ear = ear

                if ear > thresh_open:
                    self._in_blink = False
                    duration = timestamp - self._blink_start_time
                    # Real blinks: 100-400ms. Reject <60ms (noise) and >500ms (long close)
                    is_valid_duration = 0.06 < duration < 0.50
                    is_valid_depth = self._blink_peak_ear < (self.open_state_ear * 0.70)

                    if is_valid_duration and is_valid_depth:
                        self.blink_count += 1
                        self.event_times.append(timestamp)
                        detected = True
                elif (timestamp - self._blink_start_time) > 1.0:
                    # Safety: if stuck in blink for >1 second, auto-reset
                    self._in_blink = False
        else:
            # LOW reliability — abort to avoid false positives
            if self._in_blink:
                self._in_blink = False

        return {
            "blink_detected": detected,
            "count": self.blink_count,
            "pattern_score": self._score_pattern(),
            "reliability": reliability,
            "baseline": round(self.open_state_ear, 3),
            "thresh": round(thresh_close, 3),
            "source": source,
        }

    def _score_pattern(self) -> float:
        """Score blink pattern naturalness: 0.0=robotic, 1.0=natural."""
        if not self.event_times:
            return 0.5
        now = time.monotonic()
        recent = [t for t in self.event_times if now - t < 60]
        if not recent:
            return 0.2
        if len(recent) < 3:
            return 0.5

        intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        score = 1.0 if cv > 0.3 else (cv / 0.3)
        return round(float(score), 2)


# ===================================================================
# Distance Estimation
# ===================================================================

def estimate_distance(
    bbox_w: int,
    frame_w: int,
    matrix: Optional[np.ndarray] = None,
) -> float:
    """Estimate physical distance using transformation matrix or pinhole model.

    Args:
        bbox_w: Face bounding box width in pixels.
        frame_w: Frame width in pixels.
        matrix: Optional MediaPipe (4,4) transformation matrix.

    Returns:
        Estimated distance in centimeters.
    """
    if matrix is not None:
        try:
            z_cm = matrix[2, 3]
            return round(abs(float(z_cm)), 1)
        except Exception:
            pass

    # Simple pinhole camera model: D = (F * W_real) / W_pixel
    focal_length = frame_w
    real_width_cm = 14.0
    distance_cm = (focal_length * real_width_cm) / (bbox_w + 1e-10)
    return round(distance_cm, 1)


# ===================================================================
# Classification (backward compat)
# ===================================================================

def classify_face(
    fake_prob: float,
    real_prob: float,
    liveness_ok: bool,
    texture_ok: bool,
) -> tuple[str, tuple[int, int, int], str]:
    """Tiered security classification (legacy interface)."""
    if fake_prob > 0.5:
        return "CRITICAL: FAKE", (0, 0, 255), "FAKE"
    if not liveness_ok:
        return "LIVENESS FAILED", (0, 165, 255), "LIVENESS"
    if not texture_ok:
        return "FORENSIC WARNING", (0, 200, 255), "TEXTURE"
    if real_prob < 0.9:
        return "LOW CONFIDENCE", (0, 200, 255), "WARN"
    return "SHIELD: VERIFIED", (0, 255, 0), "VERIFIED"


# ===================================================================
# Backward-Compatible Aliases
# ===================================================================

def calculate_ear(landmarks, eye_indices, head_pose=(0.0, 0.0, 0.0), is_frontal=True):
    """Backward-compatible alias for compute_ear."""
    return compute_ear(landmarks, eye_indices, head_pose, is_frontal)


def analyze_blink_pattern(
    event_times: list,
    window: float = 60.0,
    window_seconds: float = None,
) -> tuple:
    """Analyze blink pattern naturalness from event timestamps.

    Returns (score, description) where score is 0.0 (robotic) to 1.0 (natural).
    """
    if window_seconds is not None:
        window = window_seconds
    if not event_times:
        return 0.2, "no_blinks"
    now = time.monotonic()
    recent = [t for t in event_times if now - t < window]
    if not recent:
        return 0.2, "no_recent_blinks"
    if len(recent) < 3:
        return 0.5, "insufficient_blinks"
    intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    cv_val = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
    score = 1.0 if cv_val > 0.3 else (cv_val / 0.3)
    score = round(float(score), 2)
    desc = "natural" if cv_val > 0.3 else "periodic"
    return score, desc


def check_texture(
    face_crop_raw: np.ndarray,
    device_baseline: Optional[float] = None,
) -> tuple:
    """Backward-compatible alias for compute_texture_score."""
    return compute_texture_score(face_crop_raw, device_baseline)
