"""
Shield-Ryzen V2 -- shield_utils package
========================================
Re-exports all utilities from shield_utils_core.py for backward
compatibility. Existing imports like `from shield_utils import X`
continue to work.

Also exposes submodules:
  - shield_utils.calibrated_decision
  - shield_utils.occlusion_detector
"""

from __future__ import annotations

# Re-export everything from the core module
from shield_utils_core import (
    # Config
    load_config,
    CONFIG,
    # Constants
    CONFIDENCE_THRESHOLD,
    BLINK_THRESHOLD,
    BLINK_TIME_WINDOW,
    LAPLACIAN_THRESHOLD,
    MEAN,
    STD,
    INPUT_SIZE,
    LEFT_EYE,
    RIGHT_EYE,
    # Logging
    setup_logger,
    # Face preprocessing
    preprocess_face,
    # EAR (new + backward compatible)
    compute_ear,
    calculate_ear,
    analyze_blink_pattern,
    # Texture (new + backward compatible)
    compute_texture_score,
    check_texture,
    _compute_hf_energy_ratio,
    # Calibration
    calibrate_device_baseline,
    ConfidenceCalibrator,
    # Classification
    classify_face,
)

from .blazeface_detector import BlazeFaceDetector

__all__ = [
    "load_config", "CONFIG",
    "CONFIDENCE_THRESHOLD", "BLINK_THRESHOLD", "BLINK_TIME_WINDOW",
    "LAPLACIAN_THRESHOLD", "MEAN", "STD", "INPUT_SIZE",
    "LEFT_EYE", "RIGHT_EYE",
    "setup_logger",
    "preprocess_face",
    "compute_ear", "calculate_ear", "analyze_blink_pattern",
    "compute_texture_score", "check_texture",
    "calibrate_device_baseline", "ConfidenceCalibrator",
    "classify_face",
    "BlazeFaceDetector",
]
