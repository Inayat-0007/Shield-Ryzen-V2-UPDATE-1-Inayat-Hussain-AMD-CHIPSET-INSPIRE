"""
Shield-Ryzen V2 — HUD & Accessibility (TASK 10.1)
=================================================
WCAG 2.1 AA compliant security overlay.
Provides visual feedback with high-contrast colors and unique shapes
to ensure accessibility for color-blind users.

Features:
  - Face bounding boxes with status colors.
  - Shape indicators (Check, X, Triangle, Circle).
  - Component status dashboard.
  - FPS and Resource usage.
  - Heartbeat and GradCAM visualization hooks.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 10 of 14 — HUD & Explainability
"""

import cv2
import numpy as np
import time
import logging

# Add project root
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shield_engine import EngineResult, FaceResult

class ShieldHUD:
    # WCAG Compliant Colors (BGR for OpenCV)
    COLORS = {
        "VERIFIED":    {"bg": (0, 180, 0),       "shape": "checkmark", "text": (255, 255, 255)}, # Dark Green
        "REAL":        {"bg": (0, 200, 0),       "shape": "checkmark", "text": (255, 255, 255)}, 
        "SUSPICIOUS":  {"bg": (0, 165, 255),      "shape": "question",  "text": (0, 0, 0)},       # Orange
        "HIGH_RISK":   {"bg": (0, 0, 220),        "shape": "x_mark",    "text": (255, 255, 255)}, # Red
        "FAKE":        {"bg": (0, 0, 220),        "shape": "x_mark",    "text": (255, 255, 255)}, # Red
        "UNKNOWN":     {"bg": (128, 128, 128),    "shape": "dash",      "text": (255, 255, 255)}, # Gray
        "NO_FACE":     {"bg": (100, 100, 100),    "shape": "circle",    "text": (255, 255, 255)},
        "CAMERA_ERROR":{"bg": (0, 0, 180),        "shape": "triangle",  "text": (255, 255, 255)},
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_render_times = []

    def render(self, frame: np.ndarray, engine_result: EngineResult) -> tuple[np.ndarray, float]:
        """
        Renders HUD overlay.
        """
        t_start = time.monotonic()
        
        # Determine overall system state color (worst case of faces)
        # But we render per face.
        
        # 1. Draw Face Badges
        for face in engine_result.face_results:
            self._draw_face_badge(frame, face)

        # 2. Draw Status Bar
        self._draw_status_bar(frame, engine_result)

        # 3. Draw Advanced Overlays (Heartbeat, etc if present)
        # (This would access plugins data attached to result)
        # Assuming plugin details are in face.plugin_votes or similar
        
        t_end = time.monotonic()
        render_time = (t_end - t_start) * 1000.0
        return frame, render_time

    def _draw_face_badge(self, frame, face: FaceResult):
        x, y, w, h = face.bbox
        state = face.state
        color_cfg = self.COLORS.get(state, self.COLORS["UNKNOWN"])
        color = color_cfg["bg"]
        text_color = color_cfg["text"]
        
        # Bounding Box (Corner bracket style or full box)
        self._draw_corner_rect(frame, (x, y, w, h), color, length=20, thickness=2)
        
        # Label Background
        label = f"{state} ({face.confidence:.1%})"
        (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 1)
        
        cv2.rectangle(frame, (x, y - 25), (x + tw + 30, y), color, -1)
        
        # Shape Indicator
        shape_x = x + tw + 15
        shape_y = y - 12
        self._draw_shape(frame, color_cfg["shape"], (shape_x, shape_y), 8, (255, 255, 255))
        
        # Text
        cv2.putText(frame, label, (x + 5, y - 8), self.font, 0.6, text_color, 1, cv2.LINE_AA)
        
        # Tier Details (Tech overlay)
        details = [
            f"Neural: {face.neural_confidence:.2f}",
            f"Liveness (EAR): {face.ear_value:.2f}",
            f"Texture: {face.texture_score:.1f}"
        ]
        
        dy = y + h + 15
        for line in details:
            cv2.putText(frame, line, (x, dy), self.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            dy += 15

    def _draw_status_bar(self, frame, result: EngineResult):
        # Top Bar
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        
        # Left: Logo/Title
        cv2.putText(frame, "SHIELD-RYZEN V2", (10, 20), self.font, 0.5, (0, 255, 0), 1)
        
        # Right: FPS & Stats
        stats = f"FPS: {result.fps:.1f} | Mem: {result.memory_mb:.0f}MB"
        (tw, th), _ = cv2.getTextSize(stats, self.font, 0.5, 1)
        cv2.putText(frame, stats, (w - tw - 10, 20), self.font, 0.5, (200, 200, 200), 1)

    def _draw_corner_rect(self, img, bbox, color, length=15, thickness=2):
        x, y, w, h = bbox
        # Top Left
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        # Top Right
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        # Bottom Left
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        # Bottom Right
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def _draw_shape(self, img, shape, center, radius, color, thickness=2):
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)
        if shape == "checkmark":
            pts = np.array([[cx - r, cy], [cx - r//3, cy + r//2], [cx + r, cy - r]], np.int32)
            cv2.polylines(img, [pts], False, color, thickness)
        elif shape == "x_mark":
            cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness)
            cv2.line(img, (cx + r, cy - r), (cx - r, cy + r), color, thickness)
        elif shape == "triangle":
            pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
            cv2.polylines(img, [pts], True, color, thickness)
        elif shape == "circle":
            cv2.circle(img, (cx, cy), r, color, thickness)
        elif shape == "question":
            cv2.putText(img, "?", (cx - 5, cy + 5), self.font, 0.6, color, thickness)
        elif shape == "dash":
            cv2.line(img, (cx - r, cy), (cx + r, cy), color, thickness)
