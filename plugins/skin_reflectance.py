"""
Shield-Ryzen V2 — Skin Reflectance Biometric Plugin (TASK 7.4)
============================================================
Analyzes light reflection properties to distinguish organic skin
from silicone masks (specular/matte) and screens (backlight/grid).

Real Skin: Subsurface scattering, specific highlight distribution.
Masks: Uniform albedo, unnatural specularity.
Screens: Moiré patterns, quantized dynamic range.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 7 of 14 — Biometric Hardening
"""

import numpy as np
import cv2
import logging

try:
    from scipy.stats import skew, kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Add project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_plugin import ShieldPlugin

class SkinReflectancePlugin(ShieldPlugin):
    name = "skin_reflectance"
    tier = "biometric"

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Analyze specular highlights and gradient distribution on face ROI.
        """
        try:
            # ROI: Cheek (usually good for texture/reflectance, avoids T-zone glare)
            x, y, w, h = face.bbox
            
            # Left Cheek ROI
            c_x = x + int(w * 0.2)
            c_y = y + int(h * 0.5)
            c_w = int(w * 0.2)
            c_h = int(h * 0.2)
            
            # Clamp
            rows, cols, _ = frame.shape
            c_y = max(0, min(c_y, rows))
            c_h = max(1, min(c_h, rows - c_y))
            c_x = max(0, min(c_x, cols))
            c_w = max(1, min(c_w, cols - c_x))
            
            roi = frame[c_y:c_y+c_h, c_x:c_x+c_w]
            
            if roi.size == 0:
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "name": self.name,
                    "explanation": "ROI invalid"
                }

            # Convert to YCrCb (Luminance)
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            y_roi = ycrcb[:, :, 0]
            
            # 1. Specular Highlight Analysis
            # Count pixels > 230 (blown out)
            highlight_ratio = np.sum(y_roi > 230) / y_roi.size
            
            # Real skin usually has small highlights (< 5%) unless sweating/oily.
            # Mask might be matte (0%) or very shiny (e.g. latex) (> 10%).
            # Screen might clip dynamic range.
            
            # 2. Gradient Analysis (Texture)
            gx = cv2.Sobel(y_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(y_roi, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            
            mean_grad = np.mean(magnitude)
            std_grad = np.std(magnitude)
            
            # Interpret
            # Real skin: Average gradients (pores/texture).
            # Smooth Mask: Low gradients.
            # Screen (Moiré): High oscillating gradients or specific pattern.
            
            verdict = "REAL"
            confidence = 0.5
            explanation = "Texture consistent with skin"
            
            if highlight_ratio > 0.25:
                verdict = "FAKE"
                confidence = 0.7
                explanation = f"Unnatural specularity ({highlight_ratio:.1%})"
            elif mean_grad < 1.5: # Very smooth — likely mask or heavy filter
                verdict = "FAKE"
                confidence = 0.6
                explanation = f"Surface too smooth (Mask/Filter? Grad={mean_grad:.1f})"
            elif mean_grad > 80.0: # Extremely noisy — likely screen Moiré
                verdict = "UNCERTAIN"
                confidence = 0.4
                explanation = f"High frequency noise (Screen/Moiré? Grad={mean_grad:.1f})"
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "name": self.name,
                "explanation": explanation,
                "metric_value": float(mean_grad)
            }

        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": str(e)}
