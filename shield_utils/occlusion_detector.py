"""
Shield-Ryzen V2 — Detailed Occlusion Detector
===============================================
Uses facial landmark positions to determine per-region visibility.
If occlusion > 30%, reduces confidence weight of liveness tier by 50%.

Checks visibility of:
  - Left eye, Right eye (critical for EAR/blink detection)
  - Nose (anchor point for alignment)
  - Mouth (lip-sync verification in Part 8)
  - Forehead (context for deepfake boundary artifacts)

Developer: Inayat Hussain | AMD Slingshot 2026
Part 2 of 12 — Face Detection, Preprocessing & Multi-Face
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

_log = logging.getLogger("OcclusionDetector")


# MediaPipe 478-mesh landmark indices for key facial regions
_REGION_INDICES = {
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 160, 158, 157, 173, 246],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385],
    "nose": [1, 2, 3, 4, 5, 6, 97, 98, 326, 327, 168, 197, 195],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 308, 13, 14],
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
}

# Occlusion threshold for reducing liveness confidence
_CONFIDENCE_REDUCTION_THRESHOLD = 0.30  # 30% occlusion
_CONFIDENCE_REDUCTION_FACTOR = 0.50     # Reduce by 50%


def detect_occlusion_detailed(
    landmarks: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    image_size: Optional[tuple[int, int]] = None,
) -> dict:
    """Analyze per-region occlusion of a detected face.

    Uses landmark positions relative to the bounding box and image
    boundaries to determine which facial regions are visible.

    Args:
        landmarks: (N, 2) array of pixel-coordinate landmarks.
            Expects N >= 400 (MediaPipe 478-mesh).
        face_bbox: (x, y, w, h) bounding box of the face.
        image_size: (width, height) of the source image. If None,
            only bbox-relative checks are performed.

    Returns:
        Dictionary with occlusion analysis:
        {
            "occlusion_percent": float,          # Overall 0.0-1.0
            "left_eye_visible": bool,
            "right_eye_visible": bool,
            "mouth_visible": bool,
            "nose_visible": bool,
            "forehead_visible": bool,
            "confidence_reduction_factor": float, # 1.0 or 0.5
            "per_region_scores": dict,            # Region->visibility
        }
    """
    result = {
        "occlusion_percent": 0.0,
        "left_eye_visible": True,
        "right_eye_visible": True,
        "mouth_visible": True,
        "nose_visible": True,
        "forehead_visible": True,
        "confidence_reduction_factor": 1.0,
        "per_region_scores": {},
    }

    if landmarks.shape[0] < 400:
        # Insufficient landmarks — assume no occlusion measurable
        _log.debug("Not enough landmarks (%d) for occlusion check", landmarks.shape[0])
        return result

    x, y, w, h = face_bbox
    region_occlusions: list[float] = []

    for region_name, indices in _REGION_INDICES.items():
        visible_count = 0
        total_count = 0

        for idx in indices:
            if idx >= landmarks.shape[0]:
                continue
            total_count += 1
            lx, ly = landmarks[idx]

            # Check 1: Landmark within extended bounding box (10% margin)
            in_bbox_x = (x - w * 0.1) <= lx <= (x + w * 1.1)
            in_bbox_y = (y - h * 0.1) <= ly <= (y + h * 1.1)

            # Check 2: Landmark within image bounds (if known)
            in_image = True
            if image_size is not None:
                img_w, img_h = image_size
                in_image = (0 <= lx < img_w) and (0 <= ly < img_h)

            if in_bbox_x and in_bbox_y and in_image:
                visible_count += 1

        # Calculate visibility ratio for this region
        visibility = visible_count / total_count if total_count > 0 else 0.0
        occlusion = 1.0 - visibility
        region_occlusions.append(occlusion)
        result["per_region_scores"][region_name] = round(visibility, 3)

        # Set per-region visibility flags
        # Consider a region "not visible" if > 40% of its landmarks are occluded
        is_visible = visibility > 0.6
        if region_name == "left_eye":
            result["left_eye_visible"] = is_visible
        elif region_name == "right_eye":
            result["right_eye_visible"] = is_visible
        elif region_name == "nose":
            result["nose_visible"] = is_visible
        elif region_name == "mouth":
            result["mouth_visible"] = is_visible
        elif region_name == "forehead":
            result["forehead_visible"] = is_visible

    # Overall occlusion is weighted average of regions
    # Eyes are weighted more heavily because EAR depends on them
    weights = {
        "left_eye": 2.0,
        "right_eye": 2.0,
        "nose": 1.5,
        "mouth": 1.0,
        "forehead": 0.5,
    }

    total_weight = 0.0
    weighted_occlusion = 0.0
    for i, (region_name, _) in enumerate(_REGION_INDICES.items()):
        w_factor = weights.get(region_name, 1.0)
        weighted_occlusion += region_occlusions[i] * w_factor
        total_weight += w_factor

    overall_occlusion = weighted_occlusion / total_weight if total_weight > 0 else 0.0
    result["occlusion_percent"] = round(overall_occlusion, 3)

    # Apply confidence reduction if occlusion exceeds threshold
    # WHY: With significant occlusion, liveness checks (EAR) are unreliable
    if overall_occlusion > _CONFIDENCE_REDUCTION_THRESHOLD:
        result["confidence_reduction_factor"] = _CONFIDENCE_REDUCTION_FACTOR
        _log.debug(
            "Occlusion %.1f%% > threshold %.0f%% -- reducing liveness confidence by 50%%",
            overall_occlusion * 100,
            _CONFIDENCE_REDUCTION_THRESHOLD * 100,
        )

    return result
