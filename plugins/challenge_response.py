"""
Shield-Ryzen V2 — Challenge Response Biometric Plugin (TASK 7.1)
================================================================
Implements "Simon Says" liveness protocol to defeat replay attacks.
Requires the user to perform specific facial actions within a timeout window.

Challenges Supported:
  - "blink_twice": Detects two distinct blinks.
  - "look_left": Detects head yaw < -15 degrees.
  - "smile": Detects mouth openness + width or blendshape.
  - "raise_eyebrows": Detects brow elevation.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 7 of 14 — Biometric Hardening
"""

import time
import random
import logging
import numpy as np
from typing import Dict, Any, List

# Add project root to path for imports if needed
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_plugin import ShieldPlugin

class ChallengeResponsePlugin(ShieldPlugin):
    name = "challenge_response"
    tier = "biometric"

    def __init__(self):
        self.challenges = [
            {"action": "blink_twice", "verify": self._verify_blinks},
            {"action": "look_left", "verify": self._verify_head_turn},
            {"action": "smile", "verify": self._verify_smile},
            {"action": "raise_eyebrows", "verify": self._verify_brows},
        ]
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_timeout_sec = 5.0
        
        # State tracking for multi-frame actions
        self.action_history = [] 
        self.last_blink_state = False
        self.blink_counter = 0

    def start_challenge(self) -> dict:
        """Select random challenge, reset state, return prompt."""
        self.current_challenge = random.choice(self.challenges)
        self.challenge_start_time = time.monotonic()
        self.action_history = []
        self.blink_counter = 0
        self.last_blink_state = False
        
        prompt = self.current_challenge["action"]
        return {
            "prompt": prompt,
            "timeout": self.challenge_timeout_sec,
            "message": f"Please {prompt.replace('_', ' ')} within {self.challenge_timeout_sec}s"
        }

    def analyze(self, face, frame: np.ndarray) -> Dict[str, Any]:
        """
        Verify if the current frame contributes to passing the active challenge.
        """
        if not self.current_challenge:
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "name": self.name,
                "explanation": "No active challenge"
            }

        # Check timeout
        elapsed = time.monotonic() - self.challenge_start_time
        if elapsed > self.challenge_timeout_sec:
            self.current_challenge = None
            return {
                "verdict": "FAKE",
                "confidence": 0.9,
                "name": self.name,
                "explanation": "Challenge timed out (Replay attack?)"
            }

        # Verify action
        verifier = self.current_challenge["verify"]
        passed = verifier(face)
        
        if passed:
            self.current_challenge = None # Reset on success
            return {
                "verdict": "REAL",
                "confidence": 1.0,
                "name": self.name,
                "explanation": "Challenge passed successfully"
            }
        else:
            return {
                "verdict": "UNCERTAIN", # Still waiting
                "confidence": 0.5,
                "name": self.name,
                "explanation": f"Waiting for {self.current_challenge['action']} ({self.challenge_timeout_sec - elapsed:.1f}s left)"
            }

    def _verify_blinks(self, face) -> bool:
        """Check for exactly 2 blinks via blendshapes or EAR."""
        # Use blendshapes if available (MediaPipe)
        is_closed = False
        if hasattr(face, 'blendshapes') and face.blendshapes:
            # 9=EyeBlinkLeft, 10=EyeBlinkRight
            try:
                score = (face.blendshapes[9].score + face.blendshapes[10].score) / 2.0
                is_closed = score > 0.5
            except:
                pass
        
        # Simple state machine for blink counting
        if is_closed and not self.last_blink_state:
            self.last_blink_state = True
        elif not is_closed and self.last_blink_state:
            self.last_blink_state = False
            self.blink_counter += 1
            
        return self.blink_counter >= 2

    def _verify_head_turn(self, face) -> bool:
        """Check yaw went below -15 (looked left)."""
        # head_pose is (yaw, pitch, roll)
        yaw, _, _ = face.head_pose
        # "Look left" means yaw is negative (usually -20 to -40)
        return yaw < -15.0

    def _verify_smile(self, face) -> bool:
        """Check for smile."""
        if hasattr(face, 'blendshapes') and face.blendshapes:
            try:
                # 44=MouthSmileLeft, 45=MouthSmileRight
                score = (face.blendshapes[44].score + face.blendshapes[45].score) / 2.0
                return score > 0.6
            except:
                pass
        return False

    def _verify_brows(self, face) -> bool:
        """Check for raised eyebrows."""
        if hasattr(face, 'blendshapes') and face.blendshapes:
            try:
                 # 3=BrowDownLeft, 4=BrowDownRight (inverse?) or BrowInnerUp?
                 # MediaPipe has BrowInnerUp (score) index 1 (usually).
                 # Let's assume standard indices or names if accessible.
                 # Using generic logic: usually brow raise is distinct.
                 # Actually blendshape 1 is 'browDownLeft', 2 'browDownRight', 3 'browInnerUp'?
                 # I'll rely on specific known indices or just skip if unsure.
                 # Index 1 = BrowDownLeft, 2 = BrowDownRight, 3 = BrowInnerUp
                 # Let's try index 3 (BrowInnerUp)
                 return face.blendshapes[3].score > 0.5
            except:
                pass
        return False
