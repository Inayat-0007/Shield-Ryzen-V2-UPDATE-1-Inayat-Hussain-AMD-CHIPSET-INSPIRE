"""
Shield-Ryzen V2 — Adversarial Patch Plugin (TASK 8.3)
=====================================================
Detects physical adversarial attacks (patches, glasses, stickers).
These attacks introduce unnatural, high-contrast patterns designed
to fool neural networks.

Real faces have smooth gradient transitions.
Adversarial patches have sharp edges and high-frequency noise.

Method:
  - Compute gradient magnitude map of face ROI.
  - Detect anomalous high-gradient clusters.
  - Flag regions with excessive edge density.

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

class AdversarialPatchPlugin(ShieldPlugin):
    name = "adversarial_patch"
    tier = "forensic"

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Analyze face for high-gradient patches.
        """
        try:
             # Use face crop
            if face.face_crop_raw is None or face.face_crop_raw.size == 0:
                 return {"verdict": "UNCERTAIN", "confidence": 0.0, "name": self.name, "explanation": "No face crop"}

            gray = cv2.cvtColor(face.face_crop_raw, cv2.COLOR_BGR2GRAY)
            
            # Compute Gradient Magnitude (Sobel)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            
            # Normalize or check absolute magnitude
            # High contrast edges have mag > 100-200.
            # Real face edges (features) are usually < 100 except eyes/nostrils.
            
            # Thresholding: "Strong Edges"
            # Raised from 150 — real face features (brows, lashes) hit 150-200
            edge_mask = mag > 250
            edge_density = np.sum(edge_mask) / mag.size
            
            # Check for Cluster Density (Patches are dense)
            # Dilate edges to connect nearby high gradients
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
            
            # Find Contours of high-gradient regions
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            significant_patches = 0
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore small noise, keep big patches (glasses/stickers)
                # Ignore huge patches (background/hair frame)
                # Face crop is ~299x299 ~ 90000 px. Patch > 5% (4500px) is huge.
                # Glasses frame -> moderate area.
                
                # Heuristic: Adversarial patches are dense noise textures.
                # If area > 1% and internal edge density is high...
                
                # Simple heuristic: If max gradient > 500 (impossible?) 255/1px Sobel creates range?
                # Sobel 3x3 scale factor implies larger values. Max possible 4*255.
                
                if area > (gray.size * 0.05): # >5% of face (raised from 2%)
                    mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    # Check internal density
                    internal_edges = np.sum(edge_mask[mask==255])
                    density = internal_edges / area
                    
                    if density > 0.5: # Very noisy/busy texture (raised from 0.3)
                        significant_patches += 1
                        if area > max_area: max_area = area

            verdict = "REAL"
            confidence = 0.5
            explanation = "No adversarial artifacts"
            
            if significant_patches >= 2:  # Need 2+ patches (1 could be glasses/eyes)
                verdict = "FAKE"
                confidence = 0.75
                explanation = f"Detected {significant_patches} high-gradient patches (Max Area {max_area/gray.size:.1%})"
                
            return {
                "verdict": verdict,
                "confidence": confidence,
                "name": self.name,
                "explanation": explanation,
                "metric_value": float(max_area)
            }

        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": str(e)}
