"""
Shield-Ryzen V2 — rPPG Heartbeat Biometric Plugin (TASK 7.2)
============================================================
Detects remote photoplethysmography (rPPG) signals from face video.
Real humans have subtle skin color changes (blood flow) at 40-180 BPM.
Video replays and masks do not.

Method:
  1. Extract green channel mean from forehead ROI
  2. Buffer signal for 5-10 seconds
  3. Analyze frequency spectrum (FFT)
  4. Check for dominant peak in 0.7-3.0 Hz range (42-180 BPM)

Developer: Inayat Hussain | AMD Slingshot 2026
Part 7 of 14 — Biometric Hardening
"""

import numpy as np
import cv2
from collections import deque
import logging

try:
    from scipy.signal import welch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Add project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_plugin import ShieldPlugin

class HeartbeatPlugin(ShieldPlugin):
    name = "heartbeat_rppg"
    tier = "biometric"

    def __init__(self, buffer_seconds: float = 5.0, fps: float = 30.0):
        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self.green_means = deque(maxlen=self.buffer_size)
        self.last_result = {"verdict": "UNCERTAIN", "confidence": 0.0, "bpm": 0}

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Process frame for rPPG signal.
        """
        try:
            # ROI: Forehead (stable skin region)
            # Use landmarks if available for precise forehead extraction
            # Fallback to bbox based crop
            x, y, w, h = face.bbox
            # Forehead is roughly top 25-35% of face height, center 50% width
            fh_h = int(h * 0.15)
            fh_y = y + int(h * 0.15)
            fh_x = x + int(w * 0.25)
            fh_w = int(w * 0.5)
            
            # Clamp to frame
            rows, cols, _ = frame.shape
            fh_y = max(0, min(fh_y, rows))
            fh_h = max(1, min(fh_h, rows - fh_y))
            fh_x = max(0, min(fh_x, cols))
            fh_w = max(1, min(fh_w, cols - fh_x))
            
            roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
            
            if roi.size == 0:
                return self.last_result

            # Mean of Green Channel (Index 1 in BGR)
            g_mean = np.mean(roi[:, :, 1])
            self.green_means.append(g_mean)

            # Need full buffer (5s) for valid FFT
            if len(self.green_means) < self.buffer_size:
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "name": self.name,
                    "explanation": f"Collecting rPPG data ({len(self.green_means)}/{self.buffer_size})"
                }

            # Process Signal
            bpm, snr = self._compute_heart_rate_fft()
            
            # Logic: Valid Human BPM is 40-180 (0.66 - 3.0 Hz)
            # SNR > 2.5 usually indicates strong periodic signal (real pulse)
            # SNR < 1.5 indicates noise (replay/mask)
            
            verdict = "UNCERTAIN"
            confidence = 0.0
            
            if 40 <= bpm <= 180 and snr > 2.0:
                verdict = "REAL"
                confidence = min(snr / 4.0, 1.0) # Map SNR 2.0-6.0 to conf 0.5-1.0
                explanation = f"Heartbeat detected: {int(bpm)} BPM (Strong Signal)"
            elif snr < 1.0:
                # Very weak signal after full buffer — likely static/replay
                # But still conservative: UNCERTAIN, not FAKE
                # Only trust this as FAKE if multiple buffers confirm
                verdict = "UNCERTAIN"
                confidence = 0.4
                explanation = f"Weak rPPG signal (SNR {snr:.2f}) — inconclusive"
            else:
                explanation = f"Noisy signal (BPM {int(bpm)}, SNR {snr:.2f})"

            self.last_result = {
                "verdict": verdict,
                "confidence": confidence,
                "name": self.name,
                "explanation": explanation,
                "metric_value": float(bpm)
            }
            return self.last_result

        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": str(e)}


    def _compute_heart_rate_fft(self):
        """Standard FFT-based heart rate estimation."""
        data = np.array(self.green_means)
        
        # Detrend (remove slow drift light changes)
        # Simple method: subtract rolling mean or linear fit
        # Using linear detrend via numpy
        x = np.arange(len(data))
        fit = np.polyfit(x, data, 1)
        detrended = data - (fit[0] * x + fit[1])
        
        # Apply Hamming window to reduce spectral leakage
        windowed = detrended * np.hamming(len(data))
        
        # FFT
        n = len(data)
        freqs = np.fft.rfft(windowed)
        magnitude = np.abs(freqs)
        frequency_axis = np.fft.rfftfreq(n, d=1.0/self.fps)
        
        # Bandpass mask (0.7 Hz to 3.0 Hz -> 42 to 180 BPM)
        mask = (frequency_axis >= 0.7) & (frequency_axis <= 3.0)
        
        if not np.any(mask):
            return 0, 0.0

        target_freqs = frequency_axis[mask]
        target_mags = magnitude[mask]
        
        # Find peak
        peak_idx = np.argmax(target_mags)
        peak_freq = target_freqs[peak_idx]
        peak_power = target_mags[peak_idx]
        
        # SNR: Peak Power / Avg Noise Power in band
        avg_power = np.mean(target_mags)
        snr = peak_power / (avg_power + 1e-6)
        
        bpm = peak_freq * 60.0
        return bpm, snr
