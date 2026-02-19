"""
Shield-Ryzen V2 — Codec Forensics Plugin (TASK 8.2)
===================================================
Detects double-compression artifacts and macroblocking.
Authentic webcam feeds (YUY2/NV12) are usually clean or singly compressed.
Deepfake streams (Virtual Camera) often re-compress (H.264/JPEG),
creating visible 8x8 block boundaries.

Method:
  - Analyze gradients at 8x8 block boundaries vs internal pixels.
  - Compute Blocking Artifact Ratio (BAR).
  - High BAR indicates heavy compression/re-encoding.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 8 of 14 — Forensic Arsenal
"""

import numpy as np
import cv2
import logging

# Add project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_plugin import ShieldPlugin

class CodecForensicsPlugin(ShieldPlugin):
    name = "codec_forensics"
    tier = "forensic"

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Detect blocking artifacts in 8x8 grid.
        """
        try:
            # Analyze full frame (or center crop) to detect stream compression
            # Not just face, as deepfake might be high quality but stream is compressed.
            
            # Use center 256x256 crop for speed
            h, w = frame.shape[:2]
            size = 256
            if w < size or h < size:
                return {"verdict": "UNCERTAIN", "confidence": 0.0, "name": self.name, "explanation": "Frame too small"}

            # Align crop to 8x8 grid (macroblock alignment)
            # Center roughly
            cy = (h // 2) // 8 * 8
            cx = (w // 2) // 8 * 8
            
            # Ensure crop fits (clamp center such that cy+/-128 is valid)
            if cy - 128 < 0: cy = 128
            if cx - 128 < 0: cx = 128
            if cy + 128 > h: cy = h - 128
            if cx + 128 > w: cx = w - 128

            crop = frame[cy-128:cy+128, cx-128:cx+128]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Compute horizontal differences
            diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
            # Vertical differences
            diff_v = np.abs(gray[1:, :] - gray[:-1, :])
            
            # 8x8 Grid Boundaries
            # Rows: 0, 8, 16... -> Diff Index: 7, 15, 23... (diff[7] is p[7]-p[8] boundary?) 
            # No, diff[7] is p[7]-p[8] if diff = p[i] - p[i+1]. Yes. 7, 15...
            
            # Boundary Mask (every 8th pixel)
            mask_h = np.zeros_like(diff_h, dtype=bool)
            mask_h[:, 7::8] = True
            
            mask_v = np.zeros_like(diff_v, dtype=bool)
            mask_v[7::8, :] = True
            
            # Calculate mean gradient at boundaries vs internal
            boundary_grad_h = np.mean(diff_h[mask_h])
            internal_grad_h = np.mean(diff_h[~mask_h])
            
            boundary_grad_v = np.mean(diff_v[mask_v])
            internal_grad_v = np.mean(diff_v[~mask_v])
            
            # Blocking Artifact Ratio
            bar_h = boundary_grad_h / (internal_grad_h + 1e-6)
            bar_v = boundary_grad_v / (internal_grad_v + 1e-6)
            
            avg_bar = (bar_h + bar_v) / 2.0
            
            # Thresholds
            # Uncompressed/High Quality: BAR ~ 1.0 (smooth crossing)
            # Compressed: BAR > 1.2 (visible steps)
            # Heavy Compression: BAR > 1.5
            
            verdict = "REAL"
            confidence = 0.5
            explanation = f"Clean/Low compression (BAR {avg_bar:.2f})"
            
            if avg_bar > 1.8: # Strong blocking
                verdict = "FAKE"
                confidence = 0.8 # High confidence it's re-streamed/compressed
                explanation = f"Double compression/Blocking artifacts (BAR {avg_bar:.2f})"
            elif avg_bar > 1.3:
                 verdict = "SUSPICIOUS" # Normal webcam MJPEG might be here
                 confidence = 0.4
                 explanation = f"Moderate compression artifacts (BAR {avg_bar:.2f})"

            return {
                "verdict": verdict,
                "confidence": confidence,
                "name": self.name,
                "explanation": explanation,
                "metric_value": float(avg_bar)
            }

        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": str(e)}
