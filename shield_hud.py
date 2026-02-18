"""
Shield-Ryzen V2 -- Premium HUD Overlay
========================================
Enterprise-grade security overlay with:
  - Glassmorphism-inspired semi-transparent panels
  - Color-coded state indicators with glow effects
  - Accessible WCAG 2.1 AA contrast ratios
  - AMD-branded professional status bar
  - Elegant center alert notifications
  - Smooth rounded-corner badge design

Developer: Inayat Hussain | AMD Slingshot 2026
"""

import os
import time
import logging
import cv2
import numpy as np
from typing import Tuple

from shield_types import EngineResult, FaceResult

_log = logging.getLogger("ShieldHUD")


class ShieldHUD:
    """Premium security overlay for video analysis.

    Provides high-contrast color indicators, accessible shapes,
    and premium visual design with glassmorphism effects.
    """

    # Premium color palette (BGR format)
    STATE_STYLES = {
        "VERIFIED": {
            "primary":   (80, 220, 60),    # Vibrant Green
            "glow":      (100, 255, 80),
            "badge_bg":  (40, 90, 30),
            "icon":      "✓",
            "shape":     "checkmark",
        },
        "REAL": {
            "primary":   (80, 220, 60),    # Same as VERIFIED initially
            "glow":      (100, 255, 80),
            "badge_bg":  (40, 90, 30),
            "icon":      "●",
            "shape":     "checkmark",
        },
        "SUSPICIOUS": {
            "primary":   (50, 180, 255),   # Warm Amber/Orange
            "glow":      (70, 200, 255),
            "badge_bg":  (30, 80, 120),
            "icon":      "?",
            "shape":     "question",
        },
        "WAIT_BLINK": {
            "primary":   (0, 165, 255),    # Orange
            "glow":      (50, 200, 255),
            "badge_bg":  (20, 60, 100),
            "icon":      "👁",
            "shape":     "circle", # Or eye shape? Circle is fine.
        },
        "HIGH_RISK": {
            "primary":   (60, 60, 230),    # Deep Red
            "glow":      (80, 80, 255),
            "badge_bg":  (60, 30, 100),
            "icon":      "✗",
            "shape":     "x_mark",
        },
        "CRITICAL": {
            "primary":   (40, 20, 255),    # Crimson
            "glow":      (60, 30, 255),
            "badge_bg":  (60, 15, 130),
            "icon":      "⚠",
            "shape":     "triangle",
        },
        "FAKE": {
            "primary":   (60, 60, 230),    # Deep Red
            "glow":      (80, 80, 255),
            "badge_bg":  (60, 30, 100),
            "icon":      "✗",
            "shape":     "x_mark",
        },
        "UNKNOWN": {
            "primary":   (160, 160, 160),  # Silver Gray
            "glow":      (190, 190, 190),
            "badge_bg":  (60, 60, 60),
            "icon":      "—",
            "shape":     "dash",
        },
        "NO_FACE": {
            "primary":   (120, 120, 120),  # Dim Gray
            "glow":      (140, 140, 140),
            "badge_bg":  (40, 40, 40),
            "icon":      "○",
            "shape":     "circle",
        },
        "CAMERA_ERROR": {
            "primary":   (40, 20, 200),
            "glow":      (60, 30, 230),
            "badge_bg":  (50, 15, 100),
            "icon":      "⚠",
            "shape":     "triangle",
        },
        "STALE_FRAME": {
            "primary":   (150, 80, 50),
            "glow":      (180, 100, 70),
            "badge_bg":  (70, 40, 30),
            "icon":      "⏳",
            "shape":     "hourglass",
        },
    }

    # Backward compat alias (tests reference self.hud.COLORS)
    COLORS = STATE_STYLES

    # Map engine states to HUD states
    STATE_MAPPING = {
        "REAL": "VERIFIED",
        "VERIFIED": "VERIFIED",
        "FAKE": "HIGH_RISK",
        "HIGH_RISK": "HIGH_RISK",
        "CRITICAL": "CRITICAL",
        "SUSPICIOUS": "SUSPICIOUS",
        "NO_FACE": "NO_FACE",
        "CAMERA_ERROR": "CAMERA_ERROR",
        "STALE_FRAME": "STALE_FRAME",
        "WAIT_BLINK": "WAIT_BLINK",
        "UNKNOWN": "UNKNOWN",
    }

    def __init__(self, use_audio: bool = False):
        self.use_audio = use_audio
        self._alert_start = 0.0
        self._alert_text = ""
        _log.info("ShieldHUD initialized")

    def render(self, frame: np.ndarray, engine_result: EngineResult) -> Tuple[np.ndarray, float]:
        """Draw premium overlay onto the provided frame."""
        t_hud_start = time.monotonic()

        if frame is None:
            return None, 0.0

        viz = frame.copy()

        # 1. Annotate Faces
        active_alert = ""
        if engine_result.face_results:
            for face in engine_result.face_results:
                self._draw_face_badge(viz, face)

                # Check for actionable alerts
                if face.advanced_info:
                    alert = face.advanced_info.get("face_alert", "")
                    if alert in ("TOO CLOSE", "TOO FAR", "POSE UNSTABLE", "DETECTION BLOCKED"):
                        active_alert = alert

        # 2. Draw Status Bar
        self._draw_status_bar(viz, engine_result)

        # 3. Draw FPS
        self._draw_fps(viz, engine_result.fps)

        # 4. Center Alert (only for actionable warnings)
        if active_alert:
            self._alert_text = active_alert
            self._alert_start = time.monotonic()

        # Show alert with fade-out (2 seconds)
        if self._alert_text:
            elapsed = time.monotonic() - self._alert_start
            if elapsed < 2.0:
                alpha = max(0.0, 1.0 - elapsed / 2.0)
                self._draw_center_alert(viz, self._alert_text, alpha)
            else:
                self._alert_text = ""

        t_hud = time.monotonic() - t_hud_start
        return viz, t_hud

    # ────────────────────────────────────────────────────────────
    #  Face Badge (Premium Design)
    # ────────────────────────────────────────────────────────────

    def _draw_face_badge(self, frame: np.ndarray, face: FaceResult):
        x, y, w, h = face.bbox
        mapped = self._get_mapped_state(face.state)
        style = self.STATE_STYLES.get(mapped, self.STATE_STYLES["UNKNOWN"])

        color = style["primary"]
        glow = style["glow"]
        badge_bg = style["badge_bg"]

        # ── Glow Border Effect ──
        # Draw outer glow (thicker, semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 2, y - 2), (x + w + 2, y + h + 2), glow, 3)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Main bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # ── Corner Accents ──
        corner_len = min(25, w // 4, h // 4)
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, 3)
        cv2.line(frame, (x, y), (x, y + corner_len), color, 3)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, 3)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, 3)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, 3)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, 3)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, 3)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, 3)

        # ── Shape Icon (Top-Right Corner) ──
        self._draw_shape(frame, style["shape"], (x + w - 28, y + 5), color)

        # ── Info Badge (Above Bounding Box) ──
        ear_val = face.ear_value
        blink_cnt = face.advanced_info.get('blinks', 0) if face.advanced_info else 0
        dist_cm = face.advanced_info.get('distance_cm', 0.0) if face.advanced_info else 0.0

        # Clean label
        state_label = mapped
        detail = f"EAR:{ear_val:.2f}  B:{blink_cnt}  {int(dist_cm)}cm"

        font = cv2.FONT_HERSHEY_SIMPLEX

        # State label (larger)
        state_scale = 0.55
        state_thick = 2
        (sw, sh), _ = cv2.getTextSize(state_label, font, state_scale, state_thick)

        # Detail label (smaller)
        detail_scale = 0.40
        detail_thick = 1
        (dw, dh), _ = cv2.getTextSize(detail, font, detail_scale, detail_thick)

        # Badge dimensions
        badge_w = max(sw, dw) + 20
        badge_h = sh + dh + 18
        badge_x = x
        badge_y = max(y - badge_h - 8, 5)

        # Semi-transparent badge background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (badge_x, badge_y),
                      (badge_x + badge_w, badge_y + badge_h),
                      badge_bg, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Badge border (thin accent)
        cv2.rectangle(frame,
                      (badge_x, badge_y),
                      (badge_x + badge_w, badge_y + badge_h),
                      color, 1)

        # Color accent bar on left side of badge
        cv2.rectangle(frame,
                      (badge_x, badge_y),
                      (badge_x + 4, badge_y + badge_h),
                      color, -1)

        # State text (outline + fill for WCAG contrast)
        text_x = badge_x + 10
        text_y1 = badge_y + sh + 5
        cv2.putText(frame, state_label, (text_x, text_y1),
                    font, state_scale, (0, 0, 0), state_thick + 1)
        cv2.putText(frame, state_label, (text_x, text_y1),
                    font, state_scale, (255, 255, 255), state_thick)

        # Detail text
        text_y2 = text_y1 + dh + 8
        cv2.putText(frame, detail, (text_x, text_y2),
                    font, detail_scale, (0, 0, 0), detail_thick + 1)
        cv2.putText(frame, detail, (text_x, text_y2),
                    font, detail_scale, (200, 210, 220), detail_thick)

    # ────────────────────────────────────────────────────────────
    #  Center Alert (Elegant Notification)
    # ────────────────────────────────────────────────────────────

    def _draw_center_alert(self, frame: np.ndarray, text: str, alpha: float = 1.0):
        """Draw an elegant, non-obtrusive center notification."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2

        # Color based on severity
        if "CLOSE" in text or "FAR" in text:
            accent = (50, 180, 255)   # Amber
            bg = (30, 80, 120)
        elif "UNSTABLE" in text:
            accent = (50, 180, 255)   # Amber
            bg = (30, 80, 120)
        elif "BLOCKED" in text:
            accent = (60, 60, 230)    # Red
            bg = (60, 30, 100)
        else:
            accent = (160, 160, 160)
            bg = (50, 50, 50)

        # Get text size
        (fw, fh), baseline = cv2.getTextSize(text, font, scale, thickness)

        # Position: lower-center (not blocking the face)
        cx = w // 2
        cy = int(h * 0.82)

        pad_x = 25
        pad_y = 12

        # Semi-transparent pill-shaped background
        overlay = frame.copy()
        x1 = cx - fw // 2 - pad_x
        y1 = cy - fh // 2 - pad_y
        x2 = cx + fw // 2 + pad_x
        y2 = cy + fh // 2 + pad_y

        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        blend = max(0.3, 0.7 * alpha)
        cv2.addWeighted(overlay, blend, frame, 1.0 - blend, 0, frame)

        # Accent border
        border_alpha = max(0.2, alpha)
        if border_alpha > 0.3:
            cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 2)
            # Left accent bar
            cv2.rectangle(frame, (x1, y1), (x1 + 4, y2), accent, -1)

        # Icon prefix
        icon = "⚠ " if ("CLOSE" in text or "FAR" in text or "BLOCKED" in text) else "◈ "

        # Text with outline
        full_text = f"{icon}{text}"
        (fw2, _), _ = cv2.getTextSize(full_text, font, scale, thickness)
        tx = cx - fw2 // 2
        ty = cy + fh // 2

        text_alpha = max(0.5, alpha)
        if text_alpha > 0.4:
            cv2.putText(frame, full_text, (tx, ty), font, scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, full_text, (tx, ty), font, scale, accent, thickness)

    # ────────────────────────────────────────────────────────────
    #  Status Bar (AMD-branded footer)
    # ────────────────────────────────────────────────────────────

    def _draw_status_bar(self, frame: np.ndarray, result: EngineResult):
        """Draw premium bottom status bar."""
        h, w = frame.shape[:2]
        bar_h = 36

        # Gradient-style background (dark with slight color tint)
        overlay = frame.copy()
        # Main bar
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 20), -1)
        # Top edge highlight
        cv2.line(overlay, (0, h - bar_h), (w, h - bar_h), (50, 50, 60), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        mapped_state = self._get_mapped_state(result.state)
        style = self.STATE_STYLES.get(mapped_state, self.STATE_STYLES["UNKNOWN"])
        state_color = style["primary"]

        # State indicator dot
        dot_x = 18
        dot_y = h - bar_h // 2
        cv2.circle(frame, (dot_x, dot_y), 5, state_color, -1)
        cv2.circle(frame, (dot_x, dot_y), 7, state_color, 1)

        # System state text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"SHIELD: {mapped_state}", (32, h - 11),
                    font, 0.50, state_color, 1, cv2.LINE_AA)

        # Camera Health (right side)
        cam = result.camera_health
        fps_val = cam.get('fps_actual', 0)
        drop_val = cam.get('drop_rate_pct', 0)

        # Color-code FPS
        fps_color = (80, 220, 80) if fps_val >= 25 else (50, 180, 255) if fps_val >= 15 else (60, 60, 230)
        cam_text = f"CAM: {fps_val:.0f} FPS"
        drop_text = f"Drop: {drop_val:.1f}%"

        # Right-aligned
        cam_w = cv2.getTextSize(cam_text, font, 0.45, 1)[0][0]
        drop_w = cv2.getTextSize(drop_text, font, 0.40, 1)[0][0]

        cv2.putText(frame, cam_text, (w - cam_w - drop_w - 25, h - 12),
                    font, 0.45, fps_color, 1, cv2.LINE_AA)
        cv2.putText(frame, drop_text, (w - drop_w - 10, h - 12),
                    font, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

        # Middle Badges (Secure Enclave + Diamond Tier)
        mid_x = w // 2 - 100
        cv2.putText(frame, "SECURE ENCLAVE", (mid_x, h - 12),
                    font, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "DIAMOND TIER", (mid_x + 130, h - 12),
                    font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  FPS Counter (Premium)
    # ────────────────────────────────────────────────────────────

    def _draw_fps(self, frame: np.ndarray, fps: float):
        """Draw a premium FPS counter with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps_text = f"{fps:.0f} FPS"
        scale = 0.55
        thick = 2

        (tw, th), _ = cv2.getTextSize(fps_text, font, scale, thick)

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (tw + 22, th + 18), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Color based on FPS
        if fps >= 25:
            color = (80, 230, 80)      # Green
        elif fps >= 15:
            color = (50, 200, 255)     # Amber
        else:
            color = (60, 80, 230)      # Red-ish

        cv2.putText(frame, fps_text, (15, th + 12),
                    font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (15, th + 12),
                    font, scale, color, thick, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  Shape Icons
    # ────────────────────────────────────────────────────────────

    def _draw_shape(self, frame: np.ndarray, shape: str, pos: Tuple[int, int],
                    color: Tuple[int, int, int]):
        """Draw accessible shape icon with glow effect."""
        x, y = pos
        size = 22

        if shape == "checkmark":
            pts = np.array([
                [x + 2, y + 12],
                [x + 8, y + 18],
                [x + 20, y + 4],
            ], dtype=np.int32)
            # Glow
            cv2.polylines(frame, [pts], False, (0, 0, 0), 5)
            cv2.polylines(frame, [pts], False, color, 3)

        elif shape == "x_mark":
            cv2.line(frame, (x + 2, y + 2), (x + size - 2, y + size - 2), (0, 0, 0), 4)
            cv2.line(frame, (x + size - 2, y + 2), (x + 2, y + size - 2), (0, 0, 0), 4)
            cv2.line(frame, (x + 2, y + 2), (x + size - 2, y + size - 2), color, 2)
            cv2.line(frame, (x + size - 2, y + 2), (x + 2, y + size - 2), color, 2)

        elif shape == "question":
            cv2.putText(frame, "?", (x + 3, y + size - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(frame, "?", (x + 3, y + size - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        elif shape == "circle":
            cv2.circle(frame, (x + size // 2, y + size // 2), size // 2, (0, 0, 0), 3)
            cv2.circle(frame, (x + size // 2, y + size // 2), size // 2, color, 2)

        elif shape == "triangle":
            pts = np.array([
                [x + size // 2, y + 2],
                [x + 2, y + size - 2],
                [x + size - 2, y + size - 2],
            ], dtype=np.int32)
            cv2.drawContours(frame, [pts], 0, (0, 0, 0), 3)
            cv2.drawContours(frame, [pts], 0, color, -1)

        elif shape == "dash":
            cv2.line(frame, (x + 3, y + size // 2), (x + size - 3, y + size // 2), (0, 0, 0), 4)
            cv2.line(frame, (x + 3, y + size // 2), (x + size - 3, y + size // 2), color, 2)

        elif shape == "hourglass":
            cv2.rectangle(frame, (x + 2, y + 2), (x + size - 2, y + size - 2), color, 1)
            cv2.line(frame, (x + 2, y + 2), (x + size - 2, y + size - 2), color, 1)
            cv2.line(frame, (x + size - 2, y + 2), (x + 2, y + size - 2), color, 1)

    # ────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────

    def _get_mapped_state(self, raw_state: str) -> str:
        return self.STATE_MAPPING.get(raw_state, "UNKNOWN")
