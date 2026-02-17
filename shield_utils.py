"""
Shield-Ryzen — Shared Utility Module
=====================================
Centralized security logic for all Shield-Ryzen engines.
Contains: face preprocessing, biometric liveness, texture analysis,
          security classification, config loading, and logging.

Developer: Inayat Hussain | AMD Slingshot 2026
"""

import cv2
import numpy as np
import math
import yaml
import os
import logging

# ─── Configuration ────────────────────────────────────────────
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def load_config(path=None):
    """Load configuration from config.yaml."""
    target = path or _config_path
    with open(target, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# ─── Constants (from config.yaml) ────────────────────────────
CONFIDENCE_THRESHOLD = CONFIG['security']['confidence_threshold']
BLINK_THRESHOLD      = CONFIG['security']['blink_threshold']
BLINK_TIME_WINDOW    = CONFIG['security']['blink_time_window']
LAPLACIAN_THRESHOLD  = CONFIG['security']['laplacian_threshold']

MEAN       = np.array(CONFIG['preprocessing']['mean'], dtype=np.float32)
STD        = np.array(CONFIG['preprocessing']['std'],  dtype=np.float32)
INPUT_SIZE = CONFIG['preprocessing']['input_size']

LEFT_EYE   = CONFIG['landmarks']['left_eye']
RIGHT_EYE  = CONFIG['landmarks']['right_eye']


# ─── Logging Setup ───────────────────────────────────────────
def setup_logger(name, level=logging.INFO):
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


# ─── Face Preprocessing (Pure NumPy) ────────────────────────
def preprocess_face(face_crop):
    """Preprocess face crop for Xception: resize 299x299, normalize to [-1, 1], NCHW.
    Pure NumPy — no PyTorch transforms needed."""
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    face_float = face_resized.astype(np.float32) / 255.0
    face_norm = (face_float - MEAN) / STD
    face_chw = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0).astype(np.float32)


# ─── Eye Aspect Ratio (Blink Detection) ─────────────────────
def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio from 6 landmark points.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Low EAR = eye closed (blink). Normal ~0.25-0.3, blink ~0.15."""
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    v1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
    v2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
    h_dist = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)

    if h_dist == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


# ─── Texture Analysis ───────────────────────────────────────
def check_texture(face_crop):
    """Laplacian variance — low value = too smooth (photo/deepfake artifact)."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ─── Security Classification ────────────────────────────────
def classify_face(fake_prob, real_prob, liveness_ok, texture_ok):
    """3-Tier Security Mode Classification.
    Returns (label, color_bgr, tier)."""
    if fake_prob > 0.50:
        return "CRITICAL: FAKE DETECTED", (0, 0, 255), "FAKE"
    elif real_prob < CONFIDENCE_THRESHOLD:
        return "WARNING: LOW CONFIDENCE", (0, 200, 255), "WARN"
    elif not liveness_ok:
        return "LIVENESS FAILED", (0, 165, 255), "LIVENESS"
    elif not texture_ok:
        return "SMOOTHNESS WARNING", (0, 200, 255), "TEXTURE"
    else:
        return "SHIELD: VERIFIED REAL", (0, 255, 0), "VERIFIED"
