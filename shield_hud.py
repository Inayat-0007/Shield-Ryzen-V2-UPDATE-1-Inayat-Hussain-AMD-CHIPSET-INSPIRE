
import os
import time
import logging
import cv2
import numpy as np
from typing import Tuple

from shield_types import EngineResult, FaceResult

_log = logging.getLogger("ShieldHUD")

class ShieldHUD:
    """Accessible security overlay for video analysis.
    
    Provides high-contrast color indicators, distinct shapes for 
    color-blind accessibility, and structured information display.
    WCAG 2.1 AA compliant.
    """

    # WCAG 2.1 AA compliant colors
    COLORS = {
        "VERIFIED":     {"bg": (0, 180, 0),     "shape": "checkmark"}, # Green
        "SUSPICIOUS":   {"bg": (0, 165, 255),    "shape": "question"},  # Orange
        "HIGH_RISK":    {"bg": (0, 0, 220),      "shape": "x_mark"},    # Red
        "CRITICAL":     {"bg": (0, 0, 255),      "shape": "triangle"}, # Pure Red
        "UNKNOWN":      {"bg": (128, 128, 128),  "shape": "dash"},      # Gray
        "NO_FACE":      {"bg": (100, 100, 100),  "shape": "circle"},    # Dim Gray
        "CAMERA_ERROR": {"bg": (0, 0, 180),      "shape": "triangle"},  # Red
        "STALE_FRAME":  {"bg": (50, 50, 150),    "shape": "hourglass"}, # Deep Purple
    }

    # Map Engine States to HUD States
    # Engine uses: "REAL", "FAKE", "NO_FACE", "CAMERA_ERROR", "STALE_FRAME"
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
        "UNKNOWN": "UNKNOWN"
    }

    def __init__(self, use_audio: bool = False):
        self.use_audio = use_audio
        _log.info("ShieldHUD initialized")

    def render(self, frame: np.ndarray, engine_result: EngineResult) -> Tuple[np.ndarray, float]:
        """Draw overlay onto the provided frame.
        
        Args:
            frame: BGR image.
            engine_result: Output from ShieldEngine.
            
        Returns:
            (annotated_frame, hud_render_time_seconds)
        """
        t_hud_start = time.monotonic()
        
        if frame is None:
             return None, 0.0

        viz = frame.copy()
        
        # 1. Annotate Faces
        active_alert = None
        if engine_result.face_results:
            # Sort by area/importance to pick primary alert?
            # Or just take the first critical one.
            for face in engine_result.face_results:
                self._draw_face_badge(viz, face)
                
                # Check for critical alerts to show centrally
                if face.advanced_info:
                    alert = face.advanced_info.get("face_alert")
                    state = self._get_mapped_state(face.state)
                    # Don't double alert if alert is just the state name, unless it's critical
                    if alert and alert not in ["VERIFIED", "REAL", "pass"]:
                         # Prioritize specific feedback
                         if alert in ["TOO CLOSE", "TOO FAR", "POSE UNSTABLE", "DETECTION BLOCKED"]:
                             active_alert = alert
                         elif state in ["HIGH_RISK", "CRITICAL", "FAKE"]:
                             active_alert = f"{state} DETECTED"

        # 2. Draw Status Bar
        self._draw_status_bar(viz, engine_result)
        
        # 3. Draw FPS
        self._draw_fps(viz, engine_result.fps)

        # 4. Central Notification (User Request)
        if active_alert:
            self._draw_central_notification(viz, active_alert)
        
        t_hud = time.monotonic() - t_hud_start
        return viz, t_hud

    def _draw_central_notification(self, frame: np.ndarray, text: str):
        """Draw a large, attention-grabbing notification in the center."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5
        thickness = 3
        
        # Determine color based on text content
        color = (0, 0, 255) # Default Red
        if "CLOSE" in text or "FAR" in text:
            color = (0, 165, 255) # Orange
        elif "UNSTABLE" in text:
            color = (0, 255, 255) # Yellow
        elif "VERIFIED" in text:
            color = (0, 255, 0)   # Green
            
        # Get text size
        (fw, fh), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        cx, cy = w // 2, h // 2
        
        # Draw background box
        pad = 20
        cv2.rectangle(frame, 
            (cx - fw // 2 - pad, cy - fh // 2 - pad), 
            (cx + fw // 2 + pad, cy + fh // 2 + pad), 
            (0, 0, 0), -1) # Black background
            
        # Draw border
        cv2.rectangle(frame, 
            (cx - fw // 2 - pad, cy - fh // 2 - pad), 
            (cx + fw // 2 + pad, cy + fh // 2 + pad), 
            color, 2)

        # Draw text
        cv2.putText(frame, text, 
            (cx - fw // 2, cy + fh // 2), 
            font, scale, color, thickness)

    def _get_mapped_state(self, raw_state: str) -> str:
        return self.STATE_MAPPING.get(raw_state, "UNKNOWN")

    def _draw_face_badge(self, frame: np.ndarray, face: FaceResult):
        x, y, w, h = face.bbox
        mapped_state = self._get_mapped_state(face.state)
        
        props = self.COLORS.get(mapped_state, self.COLORS["UNKNOWN"])
        color = props["bg"]
        shape = props["shape"]

        # Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Shape Indicator (Top Right Corner)
        self._draw_shape(frame, shape, (x + w - 25, y + 5), color)
        
        # Text Label (Top Left Corner)
        ear_val = face.ear_value
        blink_cnt = face.advanced_info.get('blinks', 0) if face.advanced_info else 0
        dist_cm = face.advanced_info.get('distance_cm', 0.0) if face.advanced_info else 0.0
        
        # Concise detail string for the badge
        detail_str = f" | EAR:{ear_val:.2f} | B:{blink_cnt} | {int(dist_cm)}cm"
        face_alert = face.advanced_info.get("face_alert") if face.advanced_info else None
        
        status_label = f"{mapped_state}"
        if face_alert and face_alert != mapped_state:
             # If alert is critical, it's shown in center, keep badge simple
             pass 

        label = f"{status_label} {detail_str}"
            
        font_scale = 0.5
        thickness = 1
        
        # Label Background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(y - 10, 20)
        cv2.rectangle(frame, (x, label_y - th - 5), (x + tw + 10, label_y + 5), (0,0,0), -1)
        
        # Text
        cv2.putText(frame, label, (x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def _draw_shape(self, frame: np.ndarray, shape: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw accessible shape icon."""
        x, y = pos
        size = 20
        # Simple geometric primitives
        if shape == "checkmark":
            # Check (poly line)
            pts = np.array([[x, y+10], [x+7, y+17], [x+20, y]], dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 3)
            # Add white outline for visibility? Or rely on color?
            cv2.polylines(frame, [pts], False, (255,255,255), 1)
            
        elif shape == "x_mark":
            # Cross
            cv2.line(frame, (x, y), (x+size, y+size), color, 3)
            cv2.line(frame, (x+size, y), (x, y+size), color, 3)
            
        elif shape == "question":
            # Just '?' char
            cv2.putText(frame, "?", (x, y+size), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        elif shape == "circle":
            cv2.circle(frame, (x+10, y+10), 10, color, 2)
            
        elif shape == "triangle":
             pts = np.array([[x+10, y], [x, y+20], [x+20, y+20]], dtype=np.int32)
             cv2.drawContours(frame, [pts], 0, color, -1) # Filled
             
        elif shape == "dash":
             cv2.line(frame, (x, y+10), (x+20, y+10), color, 3)

        elif shape == "hourglass":
             # X inside box?
             cv2.rectangle(frame, (x, y), (x+size, y+size), color, 1)
             cv2.line(frame, (x, y), (x+size, y+size), color, 1)
             cv2.line(frame, (x+size, y), (x, y+size), color, 1)

    def _draw_status_bar(self, frame: np.ndarray, result: EngineResult):
        """Draw bottom status bar with system health metrics."""
        h, w = frame.shape[:2]
        bar_h = 40
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        # Blend
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # State Text
        mapped_state = self._get_mapped_state(result.state)
        cv2.putText(frame, f"SYSTEM: {mapped_state}", (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        # Camera Health
        cam = result.camera_health
        cam_text = f"CAM: {cam.get('fps_actual', 0):.1f} FPS | Drop: {cam.get('drop_rate_pct', 0):.1f}%"
        
        # Right aligned
        text_w = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        cv2.putText(frame, cam_text, (w - text_w - 10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _draw_fps(self, frame: np.ndarray, fps: float):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
