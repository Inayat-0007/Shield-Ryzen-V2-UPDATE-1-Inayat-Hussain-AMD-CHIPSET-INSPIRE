"""
Shield-Ryzen V2 — Calibration Dataset Generator (V2)
=====================================================
Expands calibration dataset from 50 to 500+ frames with diversity:
  - Multiple skin tones (Fitzpatrick I-VI simulation)
  - Multiple lighting conditions (daylight, indoor, low, harsh, mixed)
  - Multiple camera simulations (resolutions, compression levels)
  - Both real and deepfake-simulated samples
  - Various pose angles

Uses existing 50 calibration images as seeds and augments them
with controlled transformations to simulate diversity.

Usage:
    python scripts/generate_calibration_v2.py

Developer: Inayat Hussain | AMD Slingshot 2026
Part 1 of 12 — Input Pipeline Hardening
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ─── Configuration ────────────────────────────────────────────

OUTPUT_DIR = os.path.join(_project_root, "data", "calibration_set_v2")
SEED_DIR = os.path.join(_project_root, "calibration_data")
TARGET_TOTAL = 520  # > 500 as required

# Fitzpatrick Skin Tone simulation via HSV adjustments
SKIN_TONE_ADJUSTMENTS = {
    "fitzpatrick_I":   {"brightness": 30, "saturation": -10, "hue": -5},
    "fitzpatrick_II":  {"brightness": 15, "saturation": -5, "hue": -3},
    "fitzpatrick_III": {"brightness": 0, "saturation": 0, "hue": 0},
    "fitzpatrick_IV":  {"brightness": -15, "saturation": 5, "hue": 3},
    "fitzpatrick_V":   {"brightness": -30, "saturation": 10, "hue": 5},
    "fitzpatrick_VI":  {"brightness": -45, "saturation": 15, "hue": 8},
}

LIGHTING_CONDITIONS = {
    "daylight":  {"brightness": 20, "contrast": 1.1},
    "indoor":    {"brightness": -5, "contrast": 1.0},
    "low_light": {"brightness": -40, "contrast": 0.8},
    "harsh":     {"brightness": 30, "contrast": 1.4},
    "mixed":     {"brightness": 0, "contrast": 1.2},
}

CAMERA_SIMULATIONS = {
    "hd_720p":   {"resolution": (1280, 720), "jpeg_quality": 90},
    "fhd_1080p": {"resolution": (1920, 1080), "jpeg_quality": 95},
    "vga_480p":  {"resolution": (640, 480), "jpeg_quality": 75},
    "low_qual":  {"resolution": (640, 480), "jpeg_quality": 40},
    "webcam":    {"resolution": (1280, 720), "jpeg_quality": 80},
}

COMPRESSION_LEVELS = {
    "raw": 100,
    "h264_low": 85,
    "h264_mid": 65,
    "h264_high": 40,
}

POSE_ANGLES = {
    "frontal":   0,
    "slight_left": 10,
    "left_15":   15,
    "slight_right": -10,
    "right_15":  -15,
}


# ─── Augmentation Functions ──────────────────────────────────

def _adjust_skin_tone(
    img: np.ndarray,
    brightness: int = 0,
    saturation: int = 0,
    hue: int = 0,
) -> np.ndarray:
    """Simulate different skin tones via HSV channel adjustments."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _adjust_lighting(
    img: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Simulate different lighting conditions."""
    result = img.astype(np.float32)
    result = result * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def _simulate_camera(
    img: np.ndarray,
    resolution: tuple[int, int] = (640, 480),
    jpeg_quality: int = 80,
) -> np.ndarray:
    """Simulate different camera characteristics via resize + compression."""
    # Resize to target resolution then back
    resized = cv2.resize(img, resolution)
    # Apply JPEG compression artifact
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encoded = cv2.imencode(".jpg", resized, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _simulate_pose(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Simulate slight rotation to mimic different head poses."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _simulate_deepfake(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Create a crude deepfake simulation by adding common artifacts:
    - Gaussian blur on face region (too smooth)
    - Slight color shift
    - Compression artifacts
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 2.0)
    # Blend with slight color shift
    shift = rng.randint(-15, 15, size=3)
    shifted = np.clip(blurred.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    # Heavy compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encoded = cv2.imencode(".jpg", shifted, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ─── Main Generator ──────────────────────────────────────────

def generate_calibration_dataset() -> str:
    """Generate 500+ diverse calibration frames with metadata.

    Returns the output directory path.
    """
    print("=" * 60)
    print("  SHIELD-RYZEN V2 — CALIBRATION DATASET GENERATOR")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(42)

    # Load seed images
    seed_images: list[np.ndarray] = []
    seed_dir = Path(SEED_DIR)
    if seed_dir.is_dir():
        for img_path in sorted(seed_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                seed_images.append(img)

    if not seed_images:
        print("  ⚠  No seed images found — generating fully synthetic frames")
        for i in range(10):
            frame = rng.randint(40, 220, size=(480, 640, 3), dtype=np.uint8)
            seed_images.append(frame)

    print(f"  Seed images loaded: {len(seed_images)}")

    # Metadata CSV
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    csv_file = open(metadata_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename", "seed_index", "camera_sim", "lighting",
        "skin_tone", "pose_angle", "is_real", "compression_level",
    ])

    count = 0
    skin_tones = list(SKIN_TONE_ADJUSTMENTS.items())
    lightings = list(LIGHTING_CONDITIONS.items())
    cameras = list(CAMERA_SIMULATIONS.items())
    poses = list(POSE_ANGLES.items())
    compressions = list(COMPRESSION_LEVELS.items())

    while count < TARGET_TOTAL:
        for seed_idx, seed_img in enumerate(seed_images):
            if count >= TARGET_TOTAL:
                break

            # Pick random combination
            skin_name, skin_adj = skin_tones[count % len(skin_tones)]
            light_name, light_adj = lightings[count % len(lightings)]
            cam_name, cam_cfg = cameras[count % len(cameras)]
            pose_name, pose_deg = poses[count % len(poses)]
            comp_name, comp_qual = compressions[count % len(compressions)]

            # Decide real vs fake (80% real, 20% fake)
            is_real = rng.random() > 0.2

            # Apply augmentations
            img = seed_img.copy()
            img = _adjust_skin_tone(img, **skin_adj)
            img = _adjust_lighting(img, **light_adj)
            img = _simulate_camera(img, **cam_cfg)
            img = _simulate_pose(img, pose_deg)

            if not is_real:
                img = _simulate_deepfake(img, rng)

            # Apply final compression level
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), comp_qual]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            # Save
            filename = f"calib_v2_{count:04d}.jpg"
            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, img)

            writer.writerow([
                filename, seed_idx, cam_name, light_name,
                skin_name, pose_deg, int(is_real), comp_name,
            ])

            count += 1
            if count % 100 == 0:
                print(f"  Generated: {count}/{TARGET_TOTAL}")

    csv_file.close()

    print(f"\n  ✅ Generated {count} frames → {OUTPUT_DIR}")
    print(f"  ✅ Metadata → {metadata_path}")
    print("=" * 60)

    return OUTPUT_DIR


if __name__ == "__main__":
    generate_calibration_dataset()
