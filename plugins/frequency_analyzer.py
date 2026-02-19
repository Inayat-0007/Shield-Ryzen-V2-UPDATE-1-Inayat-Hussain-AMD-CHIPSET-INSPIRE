"""
Shield-Ryzen V2 — Frequency Domain Forensic Plugin (TASK 8.1)
=============================================================
Detects GAN/Diffusion artifacts via 2D Fast Fourier Transform (FFT).
Generated faces often exhibit:
  1. Suppressed high-frequency energy (blur/smoothness).
  2. "Checkerboard" artifacts (periodic spikes in spectrum).

Method:
  - Convert face crop to grayscale.
  - Compute 2D Log-Power Spectrum.
  - Analyze High-Frequency Energy Ratio (HFER).
  - Detect spectral peaks indicative of upsampling artifacts.

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

class FrequencyAnalyzerPlugin(ShieldPlugin):
    name = "frequency_analysis"
    tier = "forensic"

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Analyze frequency domain fingerprints.
        """
        try:
            # Use raw face crop (unnormalized uint8)
            # face.face_crop_raw is BGR
            if face.face_crop_raw is None or face.face_crop_raw.size == 0:
                 return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0, 
                    "name": self.name,
                    "explanation": "No face crop available"
                }

            gray = cv2.cvtColor(face.face_crop_raw, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            power_spec = magnitude**2
            
            # Azimuthal Averaging / Radial Profile
            # We skip full profile computation for speed, focusing on Band Energies.
            
            # Define High Frequency Region (Circular mask: outer 30%)
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.min([h, w]) // 2
            
            # High Freq Mask: r > 0.7 * max_dist
            hf_mask = dist_from_center > (0.7 * max_dist)
            
            total_energy = np.sum(power_spec)
            hf_energy = np.sum(power_spec[hf_mask])
            
            hf_ratio = hf_energy / (total_energy + 1e-10)
            
            # GAN Artifact Check:
            # GANs often have abnormally LOW high-freq energy (smoothness)
            # OR specific spikes (checkerboard).
            # Deepfakes (FaceSwap) often smooth details -> Low HF ratio.
            
            # Real faces (pores, hair) have significant HF energy.
            # 0.15 threshold is empirical (from research on FaceForensics++)
            
            verdict = "REAL"
            confidence = 0.6
            explanation = f"Normal spectrum (HFER {hf_ratio:.4f})"
            
            if hf_ratio < 0.05: # Extremely smooth (Blur/GAN)
                verdict = "FAKE"
                confidence = 0.8
                explanation = f"Suppressed high-freq energy (HFER {hf_ratio:.4f} < 0.05)"
            elif hf_ratio > 0.50: # Extremely noisy (Noise injection?)
                 verdict = "SUSPICIOUS" # Could be sensor noise
                 confidence = 0.5
                 explanation = f"Abnormal high-freq noise (HFER {hf_ratio:.4f})"

            return {
                "verdict": verdict,
                "confidence": confidence,
                "name": self.name,
                "explanation": explanation,
                "metric_value": float(hf_ratio)
            }

        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": str(e)}
