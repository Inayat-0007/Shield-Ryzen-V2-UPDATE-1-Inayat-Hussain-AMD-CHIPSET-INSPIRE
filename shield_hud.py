"""
Shield-Ryzen V2 — Modern HUD Overlay (Redesigned V2)
=====================================================
Premium, responsive, app-like security overlay for real-time
deepfake detection. Fullscreen, no-overlap design with
color-coded indicators and side dashboard.

Features:
  - Glassmorphism translucent panels (no face overlap)
  - Color-coded state badges positioned ABOVE face
  - Right-side dashboard: EAR, Blinks, Distance, Tiers, Plugins
  - Alert banners for critical events
  - Animated pulse ring for REAL/VERIFIED
  - Fullscreen optimized layout

Developer: Inayat Hussain | AMD Slingshot 2026
Part 10 of 14 — HUD & Explainability (Rewrite V2)
"""

import cv2
import numpy as np
import time
import math

# ═══════════════════════════════════════════════════════════
# Color Palette (BGR for OpenCV)
# ═══════════════════════════════════════════════════════════

class Colors:
    """Modern color palette — all BGR."""
    # State colors
    VERIFIED    = (0, 220, 120)    # Emerald green
    REAL        = (0, 200, 80)     # Green
    WAIT_BLINK  = (0, 190, 255)    # Amber/Orange
    SUSPICIOUS  = (30, 160, 255)   # Orange
    HIGH_RISK   = (50, 50, 230)    # Red
    FAKE        = (40, 40, 220)    # Deep red
    CRITICAL    = (180, 40, 200)   # Magenta
    UNKNOWN     = (140, 140, 140)  # Gray
    NO_FACE     = (100, 100, 100)  # Dark gray
    CAMERA_ERR  = (30, 30, 200)    # Dark red

    # UI colors
    BG_DARK     = (18, 18, 22)     # Near-black
    BG_PANEL    = (30, 30, 38)     # Dark panel
    BG_CARD     = (42, 42, 52)     # Card bg
    TEXT_WHITE  = (240, 240, 245)  # Off-white
    TEXT_DIM    = (140, 140, 155)  # Muted text
    TEXT_BRIGHT = (255, 255, 255)  # Pure white
    ACCENT_CYAN = (220, 200, 50)   # Cyan accent
    ACCENT_BLUE = (230, 160, 40)   # Blue accent
    BORDER      = (60, 60, 70)     # Subtle border
    SUCCESS_DOT = (80, 220, 80)    # Green dot
    FAIL_DOT    = (60, 60, 220)    # Red dot
    WARN_DOT    = (40, 180, 255)   # Yellow dot

    @staticmethod
    def for_state(state: str):
        return getattr(Colors, state.upper().replace(" ", "_"), Colors.UNKNOWN)

    @staticmethod
    def state_glow(state: str):
        """Dimmed version for glow effect."""
        c = Colors.for_state(state)
        return tuple(max(0, v // 3) for v in c)


# ═══════════════════════════════════════════════════════════
# Drawing Utilities
# ═══════════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.85):
    """Draw a rounded rectangle with transparency."""
    h, w = img.shape[:2]
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(w, pt2[0]), min(h, pt2[1])
    if x2 <= x1 or y2 <= y1:
        return

    overlay = img.copy()
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_text(img, text, pos, scale=0.5, color=(240, 240, 245), thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    """Anti-aliased text with DUPLEX font for sharper rendering."""
    cv2.putText(img, str(text), pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, value, max_val, color, bg_color=(50, 50, 60)):
    """Draw a mini progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * min(1.0, value / max(max_val, 0.001)))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)


def draw_dot(img, x, y, color, radius=4):
    """Draw a small status dot."""
    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
# Plugin short-name mapping (clean display names)
# ═══════════════════════════════════════════════════════════

PLUGIN_DISPLAY_NAMES = {
    "heartbeat_rppg":       "Heartbeat",
    "skin_reflectance":     "Skin Texture",
    "frequency_analysis":   "Frequency",
    "codec_forensics":      "Codec",
    "adversarial_patch":    "Adversarial",
}


# ═══════════════════════════════════════════════════════════
# Main HUD Class
# ═══════════════════════════════════════════════════════════

class ShieldHUD:
    """Modern, fullscreen HUD overlay for Shield-Ryzen V2."""

    PANEL_WIDTH = 300   # Right-side panel width (wider to avoid truncation)
    TOP_BAR_H   = 40    # Top status bar height
    BOTTOM_BAR_H = 32   # Bottom info bar height

    # State display config
    STATE_INFO = {
        "VERIFIED":    {"label": "VERIFIED",    "desc": "Identity Confirmed"},
        "REAL":        {"label": "REAL",         "desc": "Real Human Detected"},
        "WAIT_BLINK":  {"label": "WAITING",      "desc": "Blink to Verify"},
        "SUSPICIOUS":  {"label": "SUSPICIOUS",   "desc": "Anomaly Detected"},
        "HIGH_RISK":   {"label": "HIGH RISK",    "desc": "Multiple Flags"},
        "FAKE":        {"label": "FAKE",         "desc": "Deepfake Detected"},
        "CRITICAL":    {"label": "CRITICAL",     "desc": "Attack Detected"},
        "UNKNOWN":     {"label": "SCANNING",     "desc": "Analyzing Face"},
        "NO_FACE":     {"label": "NO FACE",      "desc": "No Face Detected"},
        "CAMERA_ERROR":{"label": "CAM ERROR",    "desc": "Check Camera"},
    }

    def __init__(self, use_audio=False):
        self._start_time = time.monotonic()
        self._frame_count = 0
        self._alert_flash = 0

    def render(self, frame: np.ndarray, engine_result) -> tuple:
        """Render complete HUD overlay. Returns (frame, render_time_seconds)."""
        t0 = time.monotonic()
        self._frame_count += 1
        self._alert_flash = (self._alert_flash + 1) % 60

        h, w = frame.shape[:2]
        panel_x = w - self.PANEL_WIDTH

        # 1. TOP BAR
        self._draw_top_bar(frame, w, engine_result)

        # 2. FACE OVERLAYS (no detail text on face — just brackets + badge above)
        for face in engine_result.face_results:
            self._draw_face_overlay(frame, face, panel_x)

        # 3. RIGHT SIDE PANEL
        self._draw_side_panel(frame, panel_x, h, engine_result)

        # 4. BOTTOM BAR
        self._draw_bottom_bar(frame, w, h, engine_result)

        # 5. ALERT BANNERS (center of camera area, not over panel)
        self._draw_alerts(frame, w, h, engine_result, panel_x)

        return frame, time.monotonic() - t0

    # ═══════════════════════════════════════════════════════
    # Top Bar
    # ═══════════════════════════════════════════════════════

    def _draw_top_bar(self, frame, w, result):
        draw_rounded_rect(frame, (0, 0), (w, self.TOP_BAR_H), Colors.BG_DARK, radius=0, alpha=0.90)

        # Left: Shield logo
        draw_text(frame, "SHIELD", (14, 28), 0.7, Colors.ACCENT_CYAN, 2)
        draw_text(frame, "RYZEN V2", (110, 28), 0.55, Colors.TEXT_DIM, 1)

        # Center: LIVE indicator + face count
        pulse = abs(math.sin(time.monotonic() * 3)) > 0.5
        dot_color = (0, 0, 255) if pulse else (0, 0, 150)
        draw_dot(frame, w // 2 - 35, 20, dot_color, 6)
        draw_text(frame, "LIVE", (w // 2 - 22, 28), 0.55, Colors.TEXT_WHITE, 1)
        
        # Face count
        n_faces = len(result.face_results) if hasattr(result, 'face_results') else 0
        if n_faces > 0:
            draw_text(frame, f"| {n_faces} face{'s' if n_faces > 1 else ''}", (w // 2 + 30, 28), 0.45, Colors.TEXT_DIM, 1)

        # Right: FPS
        fps = result.fps if hasattr(result, 'fps') else 0
        fps_color = Colors.SUCCESS_DOT if fps > 20 else (Colors.WARN_DOT if fps > 10 else Colors.FAIL_DOT)
        draw_text(frame, f"{fps:.0f} FPS", (w - 100, 28), 0.55, fps_color, 2)

    # ═══════════════════════════════════════════════════════
    # Face Overlay — CLEAN (only brackets + badge, NO text on face)
    # ═══════════════════════════════════════════════════════

    def _draw_face_overlay(self, frame, face, panel_x):
        x, y, bw, bh = face.bbox
        state = face.state
        color = Colors.for_state(state)
        info = self.STATE_INFO.get(state, self.STATE_INFO["UNKNOWN"])

        # ── Corner bracket box ──
        self._draw_corner_brackets(frame, x, y, bw, bh, color, thickness=2, length=24)

        # ── Subtle glow line ──
        glow = Colors.state_glow(state)
        cv2.rectangle(frame, (x - 1, y - 1), (x + bw + 1, y + bh + 1), glow, 1, cv2.LINE_AA)

        # ── State badge — positioned ABOVE the face, clamped to stay on screen ──
        conf = face.neural_confidence if hasattr(face, 'neural_confidence') else 0
        badge_text = f" {info['label']}  {conf:.0%} "

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        badge_w = tw + 16
        badge_h = 30

        # Center badge above face, but don't let it go beyond the panel area
        badge_x = x + (bw - badge_w) // 2
        badge_x = max(4, min(badge_x, panel_x - badge_w - 8))  # Don't overlap panel
        badge_y = max(self.TOP_BAR_H + 6, y - badge_h - 10)

        draw_rounded_rect(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                          color, radius=8, alpha=0.92)
        draw_text(frame, badge_text, (badge_x + 8, badge_y + 22), 0.6, Colors.TEXT_BRIGHT, 2)

        # ── Pulse ring for REAL/VERIFIED ──
        if state in ("REAL", "VERIFIED"):
            cx, cy = x + bw // 2, y + bh // 2
            pulse_r = int(max(bw, bh) * 0.55 + 4 * abs(math.sin(time.monotonic() * 2)))
            cv2.circle(frame, (cx, cy), pulse_r, color, 1, cv2.LINE_AA)

    def _draw_corner_brackets(self, frame, x, y, w, h, color, thickness=2, length=22):
        """Draw sleek corner brackets."""
        cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════
    # Right Side Panel (Dashboard) — All details here, NOT on face
    # ═══════════════════════════════════════════════════════

    def _draw_side_panel(self, frame, panel_x, frame_h, result):
        panel_y = self.TOP_BAR_H + 2
        pw = self.PANEL_WIDTH - 4
        panel_h = frame_h - self.TOP_BAR_H - self.BOTTOM_BAR_H - 4

        # Panel background
        draw_rounded_rect(frame, (panel_x, panel_y), (panel_x + pw, panel_y + panel_h),
                          Colors.BG_PANEL, radius=10, alpha=0.85)

        cy = panel_y + 14  # cursor_y
        left = panel_x + 14
        right = panel_x + pw - 14
        content_w = pw - 28  # usable width inside panel

        # ── TITLE ──
        draw_text(frame, "ANALYSIS DASHBOARD", (left, cy + 12), 0.5, Colors.ACCENT_CYAN, 1)
        cy += 30
        cv2.line(frame, (left, cy), (right, cy), Colors.BORDER, 1)
        cy += 8

        if not result.face_results:
            draw_text(frame, "No face detected", (left, cy + 16), 0.45, Colors.TEXT_DIM, 1)
            draw_text(frame, "Position your face", (left, cy + 36), 0.4, Colors.TEXT_DIM, 1)
            draw_text(frame, "in front of camera", (left, cy + 54), 0.4, Colors.TEXT_DIM, 1)
            return

        # Show the LARGEST face in dashboard (primary target, not a small phone-screen face)
        face = max(result.face_results, key=lambda f: f.bbox[2] * f.bbox[3])
        adv = face.advanced_info if hasattr(face, 'advanced_info') and face.advanced_info else {}

        # ── OVERALL STATE ──
        state = face.state
        state_color = Colors.for_state(state)
        info = self.STATE_INFO.get(state, self.STATE_INFO["UNKNOWN"])
        draw_text(frame, info["label"], (left, cy + 16), 0.65, state_color, 2)
        draw_text(frame, info["desc"], (left, cy + 34), 0.35, Colors.TEXT_DIM, 1)
        cy += 44

        cv2.line(frame, (left, cy), (right, cy), Colors.BORDER, 1)
        cy += 10

        # ── NEURAL CONFIDENCE ──
        neural = face.neural_confidence if hasattr(face, 'neural_confidence') else 0
        draw_text(frame, "NEURAL CONFIDENCE", (left, cy + 12), 0.42, Colors.TEXT_DIM, 1)
        draw_text(frame, f"{neural:.1%}", (right - 58, cy + 12), 0.42, Colors.TEXT_WHITE, 1)
        cy += 18
        bar_color = Colors.SUCCESS_DOT if neural > 0.7 else (Colors.WARN_DOT if neural > 0.4 else Colors.FAIL_DOT)
        draw_progress_bar(frame, left, cy, content_w, 8, neural, 1.0, bar_color)
        cy += 18

        # ── EAR (Eye Aspect Ratio) ──
        ear = face.ear_value if hasattr(face, 'ear_value') else 0
        ear_rel = face.ear_reliability if hasattr(face, 'ear_reliability') else "?"
        rel_color = Colors.SUCCESS_DOT if ear_rel == "HIGH" else (Colors.WARN_DOT if ear_rel == "MEDIUM" else Colors.FAIL_DOT)

        draw_text(frame, "EYE ASPECT RATIO", (left, cy + 12), 0.42, Colors.TEXT_DIM, 1)
        draw_dot(frame, right - 8, cy + 8, rel_color, 5)
        draw_text(frame, f"{ear:.3f}", (right - 65, cy + 12), 0.42, Colors.TEXT_WHITE, 1)
        cy += 18
        draw_progress_bar(frame, left, cy, content_w, 8, ear, 0.45, Colors.ACCENT_BLUE)
        cy += 18

        # ── BLINK COUNT ──
        blink_count = adv.get("blinks", 0)
        draw_text(frame, "BLINK COUNT", (left, cy + 14), 0.42, Colors.TEXT_DIM, 1)
        blink_color = Colors.SUCCESS_DOT if blink_count > 0 else Colors.WARN_DOT
        draw_text(frame, str(blink_count), (right - 28, cy + 16), 0.65, blink_color, 2)
        cy += 30

        # ── DISTANCE ──
        dist_cm = adv.get("distance_cm", 0)
        draw_text(frame, "DISTANCE", (left, cy + 14), 0.42, Colors.TEXT_DIM, 1)
        if dist_cm > 0:
            dist_color = Colors.SUCCESS_DOT if 30 < dist_cm < 120 else Colors.WARN_DOT
            draw_text(frame, f"{dist_cm:.0f} cm", (right - 62, cy + 14), 0.48, dist_color, 1)
        else:
            draw_text(frame, "N/A", (right - 38, cy + 14), 0.48, Colors.TEXT_DIM, 1)
        cy += 30

        cv2.line(frame, (left, cy), (right, cy), Colors.BORDER, 1)
        cy += 10

        # ── TIER VERDICTS ──
        draw_text(frame, "TIER VERDICTS", (left, cy + 12), 0.45, Colors.ACCENT_CYAN, 1)
        cy += 24

        tiers = face.tier_results if hasattr(face, 'tier_results') else ("?", "?", "?")
        tier_names = ["Neural", "Liveness", "Forensic"]

        for i, (name, verdict) in enumerate(zip(tier_names, tiers)):
            v_str = str(verdict)
            is_pass = v_str.upper() in ("REAL", "PASS")
            dot_c = Colors.SUCCESS_DOT if is_pass else Colors.FAIL_DOT
            draw_dot(frame, left + 5, cy + 7, dot_c, 5)
            draw_text(frame, name, (left + 16, cy + 11), 0.42, Colors.TEXT_WHITE, 1)
            draw_text(frame, v_str, (right - 50, cy + 11), 0.42,
                      Colors.SUCCESS_DOT if is_pass else Colors.FAIL_DOT, 1)
            cy += 20

        cy += 6
        cv2.line(frame, (left, cy), (right, cy), Colors.BORDER, 1)
        cy += 10

        # ── PLUGIN STATUS ──
        draw_text(frame, "PLUGIN STATUS", (left, cy + 12), 0.45, Colors.ACCENT_CYAN, 1)
        cy += 24

        plugins = face.plugin_votes if hasattr(face, 'plugin_votes') else []
        for p in plugins:
            raw_name = p.get("name", "unknown")
            display_name = PLUGIN_DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " ").title())
            verdict = p.get("verdict", "?")

            is_ok = verdict in ("REAL", "PASS")
            is_bad = verdict == "FAKE"
            dot_c = Colors.SUCCESS_DOT if is_ok else (Colors.FAIL_DOT if is_bad else Colors.WARN_DOT)

            draw_dot(frame, left + 5, cy + 7, dot_c, 4)
            draw_text(frame, display_name, (left + 16, cy + 11), 0.42, Colors.TEXT_WHITE, 1)
            v_display = verdict[:10]
            (vw, _), _ = cv2.getTextSize(v_display, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            draw_text(frame, v_display, (right - vw - 4, cy + 11), 0.42, dot_c, 1)
            cy += 20

        cy += 6

        # ── TEXTURE SCORE ──
        tex = face.texture_score if hasattr(face, 'texture_score') else 0
        draw_text(frame, f"Texture Score: {tex:.1f}", (left, cy + 10), 0.35, Colors.TEXT_DIM, 1)

        # ── TRACKER ID ──
        tracker_id = adv.get("tracker_id", "-")
        draw_text(frame, f"Face ID: {tracker_id}", (right - 80, cy + 10), 0.35, Colors.TEXT_DIM, 1)

    # ═══════════════════════════════════════════════════════
    # Bottom Bar
    # ═══════════════════════════════════════════════════════

    def _draw_bottom_bar(self, frame, w, h, result):
        bar_y = h - self.BOTTOM_BAR_H
        draw_rounded_rect(frame, (0, bar_y), (w, h), Colors.BG_DARK, radius=0, alpha=0.88)

        # Left: Uptime
        uptime = time.monotonic() - self._start_time
        mins, secs = divmod(int(uptime), 60)
        draw_text(frame, f"Uptime: {mins:02d}:{secs:02d}", (14, h - 10), 0.4, Colors.TEXT_DIM, 1)

        # Center: Camera health
        cam_health = result.camera_health if hasattr(result, 'camera_health') else {}
        fps_status = cam_health.get("fps_status", "OK")
        status_color = Colors.SUCCESS_DOT if fps_status == "STABLE" else Colors.WARN_DOT
        draw_text(frame, f"Camera: {fps_status}", (w // 2 - 50, h - 10), 0.4, status_color, 1)

        # Center-right: Memory
        mem = getattr(result, 'memory_mb', 0)
        if not mem and hasattr(result, 'timing_breakdown'):
            perf = result.timing_breakdown.get("perf", {})
            mem = perf.get("memory_mb", 0) if isinstance(perf, dict) else 0
        if mem > 0:
            draw_text(frame, f"Mem: {mem:.0f}MB", (w // 2 + 80, h - 10), 0.4, Colors.TEXT_DIM, 1)

        # Right: Controls hint
        draw_text(frame, "Q / ESC to Exit", (w - 150, h - 10), 0.4, Colors.TEXT_DIM, 1)

    # ═══════════════════════════════════════════════════════
    # Alert Banners — centered in camera area (not over panel)
    # ═══════════════════════════════════════════════════════

    def _draw_alerts(self, frame, w, h, result, panel_x):
        alerts = []

        for face in result.face_results:
            adv = face.advanced_info if hasattr(face, 'advanced_info') and face.advanced_info else {}
            alert = adv.get("face_alert", "")
            if alert:
                alerts.append(alert)
            if face.state == "FAKE":
                alerts.append("DEEPFAKE DETECTED")
            elif face.state == "CRITICAL":
                alerts.append("ATTACK IN PROGRESS")

        if not result.face_results and hasattr(result, 'state'):
            if result.state == "CAMERA_ERROR":
                alerts.append("CAMERA ERROR")
            elif result.state == "NO_FACE":
                alerts.append("POSITION FACE IN FRAME")

        banner_y = self.TOP_BAR_H + 10
        camera_center_x = panel_x // 2  # Center of camera area (left of panel)

        for alert_text in alerts[:2]:
            flash = self._alert_flash < 35
            if not flash and "DETECTED" in alert_text:
                continue

            (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
            bw = tw + 30
            bx = camera_center_x - bw // 2
            bx = max(4, min(bx, panel_x - bw - 4))

            if "FAKE" in alert_text or "ATTACK" in alert_text:
                alert_color = Colors.FAKE
            elif "ERROR" in alert_text:
                alert_color = Colors.CAMERA_ERR
            else:
                alert_color = Colors.WARN_DOT

            draw_rounded_rect(frame, (bx, banner_y), (bx + bw, banner_y + 34),
                              alert_color, radius=8, alpha=0.90)
            draw_text(frame, alert_text, (bx + 15, banner_y + 24), 0.6, Colors.TEXT_BRIGHT, 2)
            banner_y += 42
