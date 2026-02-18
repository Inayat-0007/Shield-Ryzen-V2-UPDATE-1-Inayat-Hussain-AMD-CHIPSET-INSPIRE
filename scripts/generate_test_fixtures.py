"""
Shield-Ryzen V2 — Generate Test Fixture Images
================================================
Creates 3 synthetic face test images for test_face_pipeline.py:
  1. Clear frontal face
  2. Angled/tilted face
  3. Face with simulated sunglasses occlusion

Uses OpenCV drawing primitives to create face-like patterns
that MediaPipe can potentially detect, or that test validation
logic can exercise.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 2 of 12 — Test Infrastructure
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)


def _draw_face(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    face_w: int,
    face_h: int,
    skin_color: tuple = (180, 200, 220),  # BGR skin tone
    tilt_angle: float = 0.0,
    sunglasses: bool = False,
) -> np.ndarray:
    """Draw a stylized face onto the image using OpenCV primitives."""
    # Face oval
    cv2.ellipse(
        img,
        (center_x, center_y),
        (face_w // 2, face_h // 2),
        tilt_angle,  # rotation
        0, 360,
        skin_color,
        -1,
    )

    # Rotate offsets for tilted faces
    angle_rad = np.radians(tilt_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    def rot(dx: int, dy: int) -> tuple[int, int]:
        rx = int(cos_a * dx - sin_a * dy)
        ry = int(sin_a * dx + cos_a * dy)
        return center_x + rx, center_y + ry

    eye_w = face_w // 8
    eye_h = face_h // 12

    if sunglasses:
        # Draw sunglasses (black rectangles over eyes)
        pts_left = np.array([
            rot(-face_w // 4 - eye_w * 2, -face_h // 6 - eye_h * 2),
            rot(-face_w // 4 + eye_w * 2, -face_h // 6 - eye_h * 2),
            rot(-face_w // 4 + eye_w * 2, -face_h // 6 + eye_h * 2),
            rot(-face_w // 4 - eye_w * 2, -face_h // 6 + eye_h * 2),
        ], dtype=np.int32)
        pts_right = np.array([
            rot(face_w // 4 - eye_w * 2, -face_h // 6 - eye_h * 2),
            rot(face_w // 4 + eye_w * 2, -face_h // 6 - eye_h * 2),
            rot(face_w // 4 + eye_w * 2, -face_h // 6 + eye_h * 2),
            rot(face_w // 4 - eye_w * 2, -face_h // 6 + eye_h * 2),
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts_left], (20, 20, 20))
        cv2.fillPoly(img, [pts_right], (20, 20, 20))
        # Bridge
        bridge_start = rot(-face_w // 4 + eye_w, -face_h // 6)
        bridge_end = rot(face_w // 4 - eye_w, -face_h // 6)
        cv2.line(img, bridge_start, bridge_end, (20, 20, 20), 3)
    else:
        # Normal eyes (white with dark iris)
        left_eye = rot(-face_w // 4, -face_h // 6)
        right_eye = rot(face_w // 4, -face_h // 6)
        cv2.ellipse(img, left_eye, (eye_w, eye_h), tilt_angle, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, right_eye, (eye_w, eye_h), tilt_angle, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, left_eye, eye_w // 2, (40, 30, 20), -1)
        cv2.circle(img, right_eye, eye_w // 2, (40, 30, 20), -1)

    # Nose
    nose_tip = rot(0, face_h // 12)
    cv2.circle(img, nose_tip, face_w // 16, (150, 170, 190), -1)

    # Mouth
    mouth_center = rot(0, face_h // 4)
    mouth_w = face_w // 5
    mouth_h = face_h // 16
    cv2.ellipse(img, mouth_center, (mouth_w, mouth_h), tilt_angle, 0, 180, (100, 100, 150), 2)

    # Eyebrows
    brow_left = rot(-face_w // 4, -face_h // 4)
    brow_left_end = rot(-face_w // 8, -face_h // 4 - 5)
    brow_right = rot(face_w // 4, -face_h // 4)
    brow_right_end = rot(face_w // 8, -face_h // 4 - 5)
    cv2.line(img, brow_left, brow_left_end, (100, 80, 60), 2)
    cv2.line(img, brow_right, brow_right_end, (100, 80, 60), 2)

    return img


def generate_fixtures() -> None:
    """Generate 3 test fixture images."""
    fixtures_dir = os.path.join(_project_root, "tests", "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    # 1. Frontal face — clear, well-lit
    img1 = np.full((480, 640, 3), 200, dtype=np.uint8)  # Light background
    _draw_face(img1, 320, 240, 200, 260, skin_color=(170, 195, 215))
    cv2.imwrite(os.path.join(fixtures_dir, "frontal_face.jpg"), img1)
    print("  Created: frontal_face.jpg")

    # 2. Angled face — tilted 25 degrees
    img2 = np.full((480, 640, 3), 180, dtype=np.uint8)
    _draw_face(img2, 300, 250, 190, 250, skin_color=(155, 180, 200), tilt_angle=25.0)
    cv2.imwrite(os.path.join(fixtures_dir, "angled_face.jpg"), img2)
    print("  Created: angled_face.jpg")

    # 3. Sunglasses occlusion
    img3 = np.full((480, 640, 3), 190, dtype=np.uint8)
    _draw_face(img3, 320, 240, 200, 260, skin_color=(170, 195, 215), sunglasses=True)
    cv2.imwrite(os.path.join(fixtures_dir, "occluded_face.jpg"), img3)
    print("  Created: occluded_face.jpg")

    # 4. Multi-face image (two faces)
    img4 = np.full((480, 640, 3), 195, dtype=np.uint8)
    _draw_face(img4, 200, 240, 160, 210, skin_color=(170, 195, 215))
    _draw_face(img4, 440, 240, 140, 190, skin_color=(150, 175, 195))
    cv2.imwrite(os.path.join(fixtures_dir, "multi_face.jpg"), img4)
    print("  Created: multi_face.jpg")

    # 5. No face — plain scene
    img5 = np.full((480, 640, 3), 150, dtype=np.uint8)
    # Add some random shapes (not face-like)
    cv2.rectangle(img5, (100, 100), (300, 300), (100, 200, 100), 3)
    cv2.circle(img5, (500, 350), 50, (200, 100, 100), -1)
    cv2.imwrite(os.path.join(fixtures_dir, "no_face.jpg"), img5)
    print("  Created: no_face.jpg")

    print(f"\n  All fixtures saved to: {fixtures_dir}")


if __name__ == "__main__":
    generate_fixtures()
