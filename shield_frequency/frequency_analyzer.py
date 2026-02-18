
import numpy as np
import cv2
# Removed scipy dependency

class FrequencyAnalyzer:
    """
    Analyzes frequency domain characteristics of face crops to detect
    GAN-generated artifacts (high-frequency suppression, periodic noise).
    """

    def analyze(self, face_crop: np.ndarray) -> dict:
        """
        Analyze frequency components of a face crop.

        Args:
            face_crop: RGB or BGR image (uint8). 
                       Ideally pre-cropped aligned face.

        Returns:
            dict with frequency scores and anomaly detection.
        """
        if face_crop is None or face_crop.size == 0:
             return {"frequency_score": 0.0, "error": "Empty crop"}

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # 2. Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        # Magnitude spectrum (log scale for visualization/feature)
        magnitude = 20 * np.log(np.abs(f_shift) + 1e-8)
        
        # 3. Analyze Frequency Bands (Radial Profile)
        # Calculate energy at different radii from center (DC)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x**2 + y**2)
        r = r.astype(int)
        
        # Sum energy (magnitude) for each radius
        # Typically Real faces have smooth 1/f decay.
        # GANs drop off sharply at high frequencies or have spikes.
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)

        # 4. Features
        # High Frequency Ratio: Energy in outer 30% vs inner 30%
        limit = len(radial_profile)
        low_band = radial_profile[:int(limit * 0.3)]
        high_band = radial_profile[int(limit * 0.7):]
        
        low_energy = np.mean(low_band) if len(low_band) > 0 else 1.0
        high_energy = np.mean(high_band) if len(high_band) > 0 else 0.0
        
        hf_ratio = high_energy / low_energy if low_energy > 0 else 0.0
        
        # 5. Moire / Screen Detection (Periodic Spikes)
        # Real skin has smooth radial profile. Screens have spikes.
        delta = np.abs(np.diff(radial_profile))
        # Skip first few bins (DC region) — the DC-to-first-harmonic transition
        # always creates a large spike that would falsely trigger moire detection
        delta_no_dc = delta[3:] if len(delta) > 3 else delta
        max_spike = np.max(delta_no_dc) if len(delta_no_dc) > 0 else 0.0
        moire_detected = max_spike > 30.0 # Shield against intense digital display grids
        
        # High-freq energy variance (Real skin is chaotic, screens are ordered)
        spectral_variance = np.var(radial_profile[int(limit * 0.5):])
        order_anomaly = spectral_variance > 200.0 # High order suggests monitor re-scan patterns

        # Detect Anomalies
        spectral_anomaly = (hf_ratio < 0.003) or moire_detected or order_anomaly
        
        # Frequency Score (0=Fake/GAN/Screen, 1=Real)
        score = min(1.0, hf_ratio * 10.0) 
        if moire_detected or order_anomaly:
            score *= 0.1 # Severe penalty for display artifacts
        
        return {
            "frequency_score": round(score, 3),
            "high_freq_ratio": round(hf_ratio, 4),
            "spectral_anomaly": bool(spectral_anomaly),
            "moire_detected": bool(moire_detected),
            "dominant_frequencies": radial_profile[:10].tolist(),
            "explanation": "High frequency suppression" if (hf_ratio < 0.02) else ("Screen/Moire artifacts" if moire_detected else "Natural spectral decay")
        }
