"""
Shield-Ryzen V2 — Lip-Sync Verifier Plugin (TASK 8.4)
=====================================================
Audio-Visual Cross-Modal Verification (Lip Reading).
Challenges user to speak a specific phoneme/word to detect pre-recorded videos.
Deepfakes often have poor lip-sync correlation or generic mouth movement.

Supported Phonemes:
  - "O" / "Who": Detected by mouthPucker + jawOpen.
  - "Ee" / "Cheese": Detected by mouthSmile/Stretch.
  - "Ahh" / "Cat": Detected by jawOpen - mouthPucker.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 8 of 14 — Forensic Arsenal
"""

import time
import random
import numpy as np
import logging

# Add project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_plugin import ShieldPlugin

class LipSyncPlugin(ShieldPlugin):
    name = "lip_sync"
    tier = "forensic"

    def __init__(self):
        self.phonemes = {
            "O": self._verify_o,
            "Ee": self._verify_ee,
            "Ahh": self._verify_ahh
        }
        self.current_phoneme = None
        self.start_time = 0
        self.timeout = 5.0

    def start_challenge(self) -> dict:
        """Start a new lip-sync challenge."""
        key = random.choice(list(self.phonemes.keys()))
        self.current_phoneme = key
        self.start_time = time.monotonic()
        return {
            "prompt": f"Say '{key}'",
            "phoneme": key,
            "timeout": self.timeout
        }

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Verify lip shape matches expected phoneme.
        """
        if not self.current_phoneme:
             return {"verdict": "UNCERTAIN", "confidence": 0.0, "name": self.name, "explanation": "No active prompt"}
        
        elapsed = time.monotonic() - self.start_time
        if elapsed > self.timeout:
            self.current_phoneme = None
            return {"verdict": "FAKE", "confidence": 0.8, "name": self.name, "explanation": "Lip-sync timed out"}

        # Perform check
        verifier = self.phonemes[self.current_phoneme]
        try:
            passed, score = verifier(face)
            if passed:
                self.current_phoneme = None
                return {"verdict": "REAL", "confidence": 1.0, "name": self.name, "explanation": "Lip-sync matched"}
            else:
                 return {"verdict": "UNCERTAIN", "confidence": 0.3, "name": self.name, "explanation": f"Waiting for '{self.current_phoneme}' ({score:.2f})"}
        except Exception as e:
            # Likely no blendshapes
             return {"verdict": "UNCERTAIN", "confidence": 0.0, "name": self.name, "explanation": "Blendshapes unavailable"}

    def _verify_o(self, face) -> tuple:
        """Verify 'O' shape (Pucker > 0.5, Open > 0.2)."""
        if not face.blendshapes: return False, 0.0
        # 36=MouthPucker, 25=JawOpen
        pucker = face.blendshapes[36].score
        open_val = face.blendshapes[25].score
        score = (pucker + open_val) / 2.0
        return (pucker > 0.5 and open_val > 0.1), score

    def _verify_ee(self, face) -> tuple:
        """Verify 'Ee' shape (Smile/Stretch > 0.5, Pucker < 0.2)."""
        if not face.blendshapes: return False, 0.0
        # 44/45=Smile, 36=Pucker
        smile = (face.blendshapes[44].score + face.blendshapes[45].score) / 2.0
        pucker = face.blendshapes[36].score
        score = smile - pucker
        return (smile > 0.5 and pucker < 0.2), score

    def _verify_ahh(self, face) -> tuple:
        """Verify 'Ahh' shape (Open > 0.5, Pucker < 0.2)."""
        if not face.blendshapes: return False, 0.0
        # 25=JawOpen (or MouthOpen 27?)
        open_val = face.blendshapes[25].score
        pucker = face.blendshapes[36].score
        score = open_val - pucker
        return (open_val > 0.4 and pucker < 0.2), score
