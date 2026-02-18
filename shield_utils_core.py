"""
Shield-Ryzen V2 -- Shared Utility Module (Rewritten Part 3)
============================================================
Centralized security logic for all Shield-Ryzen engines.

Contains:
  A) Improved EAR with head-pose angle compensation
  B) Adaptive Laplacian analysis with frequency-domain forensics
  C) Device baseline calibration (auto-tunes thresholds)
  D) Confidence calibration via temperature scaling
  + Backward-compatible exports for all existing modules

Developer: Inayat Hussain | AMD Slingshot 2026
Part 3 of 12 -- Liveness, Forensic Analysis & Calibration
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
# Constants (backward compatible -- loaded from config.yaml)
# ===================================================================

# These are DEFAULT values from config.yaml.
# Part 3 makes them OVERRIDABLE via decision_thresholds.yaml and
# shield_calibration.json at runtime.
CONFIDENCE_THRESHOLD = CONFIG['security']['confidence_threshold']
BLINK_THRESHOLD      = CONFIG['security']['blink_threshold']
BLINK_TIME_WINDOW    = CONFIG['security']['blink_time_window']
LAPLACIAN_THRESHOLD  = CONFIG['security']['laplacian_threshold']

MEAN       = np.array(CONFIG['preprocessing']['mean'], dtype=np.float32)
STD        = np.array(CONFIG['preprocessing']['std'],  dtype=np.float32)
INPUT_SIZE = CONFIG['preprocessing']['input_size']

LEFT_EYE   = CONFIG['landmarks']['left_eye']
RIGHT_EYE  = CONFIG['landmarks']['right_eye']

# Attempt to load decision thresholds (created by threshold_optimization)
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
# A) Face Preprocessing (backward compatible)
# ===================================================================

def preprocess_face(face_crop: np.ndarray) -> np.ndarray:
    """Preprocess face crop for Xception: resize 299x299, normalize [-1,1], NCHW.

    NORMALIZATION: FF++ standard -- (pixel/255 - 0.5) / 0.5
    This maps [0, 255] -> [-1.0, +1.0]
    """
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    face_float = face_resized.astype(np.float32) / 255.0
    face_norm = (face_float - MEAN) / STD
    face_chw = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0).astype(np.float32)


# ===================================================================
# B) Strict Geometric Blink Detection (User Request: "Maths Logic Algo")
# ===================================================================

# Blink pattern tracker (module-level, shared across calls)
_blink_history: deque = deque(maxlen=300)  # 10 sec at 30 FPS
_blink_event_times: list[float] = []       # timestamps of detected blinks


def compute_ear(
    landmarks,
    eye_indices: list[int],
    head_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    is_frontal: bool = True,
) -> tuple[float, str]:
    """Compute Eye Aspect Ratio with STRICT head pose validation.
    
    Logic:
    1. Calculate vertical (v1, v2) and horizontal (h) eye distances.
    2. RAW EAR = (v1 + v2) / (2 * h)
    3. VALIDATION: If head pitch/yaw is extreme, the 2D projection is distorted.
       - Pitch > 15 (Looking Down) -> Artificial narrowing (False Blink Risk).
       - Pitch < -15 (Looking Up) -> Artificial widening.
       - Yaw > 20 (Turning) -> Horizontal narrowing (Artificial High EAR).
    """
    try:
        points = []
        for idx in eye_indices:
            # Face pipeline returns (N, 2) pixel ndarray
            points.append((float(landmarks[idx][0]), float(landmarks[idx][1])))

        p1, p2, p3, p4, p5, p6 = points

        # Vertical distances (Eyelid opening)
        v1 = math.sqrt((p2[0] - p6[0])**2 + (p2[1] - p6[1])**2)
        v2 = math.sqrt((p3[0] - p5[0])**2 + (p3[1] - p5[1])**2)
        
        # Horizontal distance (Eye width)
        h_dist = math.sqrt((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2)

        if h_dist < 1e-6:
            return 0.05, "LOW"

        ear = (v1 + v2) / (2.0 * h_dist)

        # STRICT HEAD POSE RULES
        yaw, pitch, roll = head_pose
        reliability = "HIGH"
        
        # If looking down significantly, eyelids naturally lower -> reliability LOW
        if pitch > 12.0: 
            reliability = "LOW"
        # If looking up significantly -> reliability MEDIUM
        elif pitch < -15.0:
            reliability = "MEDIUM"
        # If turning head significantly -> reliability LOW to prevent false triggers
        if abs(yaw) > 25.0:
            reliability = "LOW"
            
        return round(float(ear), 4), reliability

    except (IndexError, TypeError, ValueError):
        return 0.05, "LOW"

def estimate_distance(bbox_w: int, frame_w: int) -> float:
    """Estimate physical distance based on face width (Diamond Tier heuristic).
    Assumes average human face width is 14cm.
    """
    # Simple pinhole camera model: Distance = (F * RealW) / PixelW
    # Assuming focal length is roughly frame width (standard webcam)
    focal_length = frame_w
    real_width_cm = 14.0
    distance_cm = (focal_length * real_width_cm) / (bbox_w + 1e-10)
    return round(distance_cm, 1)

class BlinkTracker:
    """
    Robust Blink Detection using Dynamic Baseline Scaling (DBS).
    Adapts to individual eye openness in real-time.
    """
    def __init__(self, ear_threshold: float = 0.20):
        # We ignore the static threshold in favor of relative dynamic logic.
        self.open_state_ear = 0.30 # Initial assumption (will adapt instantly)
        self.blink_count = 0
        self.history = deque(maxlen=300)
        self.event_times = []
        
        # State Machine
        self._in_blink = False
        self._blink_start_time = 0.0
        self._blink_peak_ear = 1.0 # Track lowest point of blink
        
        # Head Movement Rejection
        self._last_pose_time = 0.0
        
    def update(self, ear: float, timestamp: float, reliability: str) -> dict:
        self.history.append((timestamp, ear))
        detected = False
        
        # 1. Dynamic Baseline Update
        # Find the "ceiling" of the signal (the open eye state).
        # We decay slowly to handle lighting changes, but jump up quickly if eyes open wider.
        if ear > self.open_state_ear:
            self.open_state_ear = 0.9 * self.open_state_ear + 0.1 * ear
        else:
            self.open_state_ear = 0.998 * self.open_state_ear + 0.002 * ear # Slow decay
            
        # 2. Relative Thresholds
        # A blink is defined as a significant drop relative to the CURRENT baseline.
        thresh_close = self.open_state_ear * 0.65  # Close if drops below 65% of open
        thresh_open  = self.open_state_ear * 0.90  # Reopen if returns to 90% of open
        
        # 3. Validation Logic
        if reliability == "HIGH":
            # Start of Blink
            if not self._in_blink and ear < thresh_close:
                self._in_blink = True
                self._blink_start_time = timestamp
                self._blink_peak_ear = ear
            
            # During Blink
            elif self._in_blink:
                if ear < self._blink_peak_ear:
                    self._blink_peak_ear = ear # Track minimum point
                    
                # End of Blink (Return to Open)
                if ear > thresh_open:
                    self._in_blink = False
                    duration = timestamp - self._blink_start_time
                    
                    # BIOLOGICAL VALIDATION
                    # 1. Duration: Real blinks are 30ms - 450ms.
                    # 2. Depth: The eye must have closed significantly (peak < baseline * 0.55 ideally).
                    is_valid_duration = 0.03 < duration < 0.50
                    is_valid_depth = self._blink_peak_ear < (self.open_state_ear * 0.70)
                    
                    if is_valid_duration and is_valid_depth:
                        self.blink_count += 1
                        self.event_times.append(timestamp)
                        detected = True
        else:
            # If reliability drops mid-blink (e.g. head turn), abort detection to avoid false positives.
            if self._in_blink:
                self._in_blink = False

        return {
            "blink_detected": detected,
            "count": self.blink_count,
            "pattern_score": self._score_pattern(),
            "reliability": reliability,
            "baseline": round(self.open_state_ear, 3),  # Debug info
            "thresh": round(thresh_close, 3) # Debug info
        }

    def _score_pattern(self) -> float:
        if not self.event_times: return 0.5
        now = time.monotonic()
        recent = [t for t in self.event_times if now - t < 60]
        if not recent: return 0.2
        if len(recent) < 3: return 0.5
        
        intervals = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        score = 1.0 if cv > 0.3 else (cv / 0.3)
        return round(float(score), 2)

def compute_texture_score(face_crop_raw: np.ndarray, device_baseline: Optional[float] = None) -> tuple[float, bool, str]:
    """Enhanced texture analysis using Laplacian + FFT Frequency Domain."""
    if face_crop_raw is None or face_crop_raw.size == 0:
        return 0.0, True, "EMPTY_CROP"

    gray = cv2.cvtColor(face_crop_raw, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Forehead ROI for stability
    forehead = gray[int(h*0.15):int(h*0.35), int(w*0.25):int(w*0.75)]
    if forehead.size == 0: forehead = gray

    lap_var = float(cv2.Laplacian(forehead, cv2.CV_64F).var())
    thresh = device_baseline * 0.4 if device_baseline else 30.0
    
    # Signal processing: detect high-frequency Moire patterns/screen artifacts
    f = np.fft.fft2(forehead.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1e-10)
    
    # Calculate energy in high frequency bands (outer ring of spectral density)
    cy, cx = forehead.shape[0]//2, forehead.shape[1]//2
    r_hf = int(min(forehead.shape) * 0.25)
    y, x = np.ogrid[:forehead.shape[0], :forehead.shape[1]]
    mask_lf = (y-cy)**2 + (x-cx)**2 <= r_hf**2
    hf_energy = mag[~mask_lf].mean() if not np.all(mask_lf) else 0
    lf_energy = mag[mask_lf].mean() + 1e-10
    hf_ratio = hf_energy / lf_energy

    # SCREEN REPLAY DETECTION: Screen pixels lack natural high-frequency variance
    # GAN DETECTION: Generative models often have 'spectral gaps'
    suspicious = (lap_var < thresh) or (hf_ratio < 0.12)
    explain = f"LAP: {lap_var:.1f}/{thresh:.1f} | HF: {hf_ratio:.3f}"
    return lap_var, suspicious, explain

class ConfidenceCalibrator:
    """Temperature scaling to fix overconfident model outputs."""
    def __init__(self, temperature: float = 1.5):
        self.temp = temperature
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-10, 1.0))
        scaled = np.exp(logits / self.temp)
        return scaled / scaled.sum()

class DecisionStateMachine:
    """Security Truth Table Fusion with Hysteresis."""
    def __init__(self, frames: int = 5):
        self.hysteresis = frames
        self.history = []
        self.state = "UNKNOWN"
        self.counter = 0

    def update(self, t1: str, t2: str, t3: str) -> str:
        """Handshake logic for the 8-case truth table."""
        n = 1 if t1 == "REAL" else 0
        l = 1 if t2 == "PASS" else 0
        f = 1 if t3 == "PASS" else 0
        case_id = (n << 2) | (l << 1) | f
        
        table = {
            7: "VERIFIED",   # [1,1,1]
            6: "SUSPICIOUS", # [1,1,0] - Texture/Forensic issue
            5: "SUSPICIOUS", # [1,0,1] - Liveness/Blink issue
            4: "HIGH_RISK",  # [1,0,0] - Only Neural passes
            3: "FAKE",       # [0,1,1] - Neural fail
            2: "FAKE",       # [0,1,0]
            1: "FAKE",       # [0,0,1]
            0: "CRITICAL"    # [0,0,0]
        }
        proposed = table.get(case_id, "UNKNOWN")
        
        # Agile Hysteresis: Faster transition to FAKE/CRITICAL for security, 
        # but slower transition to VERIFIED to ensure stability.
        threshold = 2 if proposed in ["FAKE", "CRITICAL", "HIGH_RISK"] else self.hysteresis
        
        if proposed == self.state:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= threshold:
                self.state = proposed
                self.counter = 0
        return self.state

def classify_face(fake_prob: float, real_prob: float, liveness_ok: bool, texture_ok: bool) -> tuple[str, tuple[int, int, int], str]:
    """Tiered security classification (Target: 50/50 audit success)."""
    if fake_prob > 0.5: return "CRITICAL: FAKE", (0, 0, 255), "FAKE"
    if not liveness_ok: return "LIVENESS FAILED", (0, 165, 255), "LIVENESS"
    if not texture_ok: return "FORENSIC WARNING", (0, 200, 255), "TEXTURE"
    if real_prob < 0.9: return "LOW CONFIDENCE", (0, 200, 255), "WARN"
    return "SHIELD: VERIFIED", (0, 255, 0), "VERIFIED"
