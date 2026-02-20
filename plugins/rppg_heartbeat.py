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

    def __init__(self, buffer_seconds: float = 5.0, fps: float = 15.0):
        self.nominal_fps = fps
        self.buffer_size = int(buffer_seconds * fps)  # 75 at 15fps
        self.green_means = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        self.last_result = {"verdict": "UNCERTAIN", "confidence": 0.0, "name": self.name, "explanation": "Initializing rPPG..."}
        self._analysis_count = 0

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Process frame for rPPG signal.
        Uses actual timestamps for accurate sampling rate estimation.
        """
        try:
            import time as _time
            # ROI: Forehead (stable skin region)
            x, y, w, h = face.bbox
            # Forehead is roughly top 25-35% of face height, center 50% width
            fh_h = int(h * 0.15)
            fh_y = y + int(h * 0.15)
            fh_x = x + int(w * 0.25)
            fh_w = int(w * 0.5)
            
            # Clamp to frame
            rows, cols = frame.shape[:2]
            fh_y = max(0, min(fh_y, rows))
            fh_h = max(1, min(fh_h, rows - fh_y))
            fh_x = max(0, min(fh_x, cols))
            fh_w = max(1, min(fh_w, cols - fh_x))
            
            roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
            
            if roi.size == 0:
                return self.last_result

            # Mean of Green Channel (Index 1 in BGR)
            g_mean = np.mean(roi[:, :, 1])
            now = _time.monotonic()
            self.green_means.append(g_mean)
            self.timestamps.append(now)

            # Need at least 60 samples (~4s at 15fps) for valid FFT
            MIN_SAMPLES = 60
            if len(self.green_means) < MIN_SAMPLES:
                progress = len(self.green_means)
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "name": self.name,
                    "explanation": f"Buffering rPPG ({progress}/{MIN_SAMPLES})",
                    "metric_value": 0.0
                }

            # Estimate actual sampling rate from timestamps
            ts_array = np.array(self.timestamps)
            dt = np.diff(ts_array)
            actual_fps = 1.0 / (np.median(dt) + 1e-10)
            # Clamp to sane range
            actual_fps = max(5.0, min(actual_fps, 60.0))

            # Process Signal
            bpm, snr = self._compute_heart_rate_fft(actual_fps)
            self._analysis_count += 1
            
            verdict = "UNCERTAIN"
            confidence = 0.0
            
            if 40 <= bpm <= 180 and snr > 2.0:
                verdict = "REAL"
                confidence = min(snr / 4.0, 1.0)
                explanation = f"Heartbeat detected: {int(bpm)} BPM (SNR {snr:.1f})"
            elif snr < 1.0 and self._analysis_count > 5:
                # Very weak signal after multiple analyses — likely static/replay
                verdict = "UNCERTAIN"
                confidence = 0.4
                explanation = f"Weak rPPG signal (SNR {snr:.2f}) — inconclusive"
            else:
                explanation = f"Analyzing (BPM {int(bpm)}, SNR {snr:.1f}, fps={actual_fps:.0f})"

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


    def _compute_heart_rate_fft(self, actual_fps=None):
        """Standard FFT-based heart rate estimation with actual sampling rate."""
        data = np.array(self.green_means)
        fps = actual_fps or self.nominal_fps
        
        # Detrend (remove slow drift light changes)
        x = np.arange(len(data))
        fit = np.polyfit(x, data, 1)
        detrended = data - (fit[0] * x + fit[1])
        
        # Apply Hamming window to reduce spectral leakage
        windowed = detrended * np.hamming(len(data))
        
        # FFT
        n = len(data)
        freqs = np.fft.rfft(windowed)
        magnitude = np.abs(freqs)
        frequency_axis = np.fft.rfftfreq(n, d=1.0/fps)
        
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

