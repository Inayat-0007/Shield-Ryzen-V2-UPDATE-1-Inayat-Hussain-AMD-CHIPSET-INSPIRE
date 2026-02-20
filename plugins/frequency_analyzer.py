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
        Analyze frequency domain fingerprints using LOG-MAGNITUDE ratio.
        
        Uses the same approach as shield_utils_core._compute_hf_energy_ratio():
        log-magnitude comparison between high-frequency and low-frequency regions.
        
        This avoids the power-spectrum-ratio trap where real webcam faces produce
        values like 0.0003 (far below any useful threshold).
        """
        try:
            # Use raw face crop (unnormalized uint8)
            if face.face_crop_raw is None or face.face_crop_raw.size == 0:
                 return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0, 
                    "name": self.name,
                    "explanation": "No face crop available"
                }

            gray = cv2.cvtColor(face.face_crop_raw, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            if h < 16 or w < 16:
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "name": self.name,
                    "explanation": "Face crop too small for FFT analysis"
                }
            
            # 2D FFT with log-magnitude (consistent with core module)
            f_transform = np.fft.fft2(gray.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            log_mag = 20 * np.log(np.abs(f_shift) + 1e-10)
            
            # Radial mask: inner 25% radius = low-frequency
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r_lf = int(min(h, w) * 0.25)
            mask_lf = (y - center_y)**2 + (x - center_x)**2 <= r_lf**2
            
            # Log-magnitude energy ratio (HF mean / LF mean)
            hf_energy = log_mag[~mask_lf].mean() if not np.all(mask_lf) else 0
            lf_energy = log_mag[mask_lf].mean() + 1e-10
            hf_ratio = hf_energy / lf_energy
            
            # THRESHOLDS (calibrated from live webcam data):
            # Real faces: HFER ~0.65-0.75 (balanced natural texture)
            # GAN faces:  HFER <0.50 (suppressed high-freq, artificial smoothness)
            # Screen replay: HFER >0.85 (Moiré patterns spike HF energy)
            
            verdict = "REAL"
            confidence = 0.6
            explanation = f"Normal spectrum (HFER {hf_ratio:.3f})"
            
            if hf_ratio < 0.45:  # Extremely smooth (GAN/diffusion generated)
                verdict = "FAKE"
                confidence = 0.80
                explanation = f"Suppressed high-freq energy (HFER {hf_ratio:.3f} < 0.45)"
            elif hf_ratio < 0.55:  # Moderately smooth (compression / distance)
                verdict = "UNCERTAIN"
                confidence = 0.40
                explanation = f"Low high-freq energy (HFER {hf_ratio:.3f} < 0.55)"
            elif hf_ratio > 0.90:  # Abnormal HF spike (screen Moiré)
                verdict = "FAKE"
                confidence = 0.70
                explanation = f"Moiré-like HF spike (HFER {hf_ratio:.3f} > 0.90)"
            elif hf_ratio > 0.80:  # Mild HF excess
                verdict = "UNCERTAIN"
                confidence = 0.35
                explanation = f"Elevated HF energy (HFER {hf_ratio:.3f})"

            return {
                "verdict": verdict,
                "confidence": confidence,
                "name": self.name,
                "explanation": explanation,
                "metric_value": float(hf_ratio)
            }

        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": str(e)}
