"""
Shield-Ryzen V2 — Plugin Architecture Interface (TASK 6.2)
==========================================================
Defines the `ShieldPlugin` base class for modular detection.
Parts 7-8 (rPPG, Depth, Audio, Temporal) will implement this interface.

Engine Integration:
  - ShieldEngine loads plugins
  - Each frame, plugins analyze `FaceDetection` and `Frame`
  - Returns `PluginVote` structured dictionary
  - DecisionStateMachine fuses all votes

Developer: Inayat Hussain | AMD Slingshot 2026
Part 6 of 14 — Integration & Security
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class ShieldPlugin(ABC):
    """
    Abstract Base Class for all Shield-Ryzen detection plugins.
    Ensures modularity and strict voting protocol.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the plugin (e.g., 'rPPG-BVP')."""
        pass

    @property
    @abstractmethod
    def tier(self) -> str:
        """
        Security tier classification:
          - 'biometric': Checks biological signals (Pulse, Blinking, Gaze)
          - 'forensic': Checks low-level artifacts (Texture, Noise, Frequency)
          - 'neural': Checks high-level semantics (Deepfakes, anomalies)
          - 'temporal': Checks consistency over time
        """
        pass

    @abstractmethod
    def analyze(self, face_data: Any, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run detection analysis on a single face.

        Args:
            face_data: `FaceDetection` object (from shield_face_pipeline)
            frame: Full BGR frame (numpy array)

        Returns:
            Dictionary containing vote:
            {
                "verdict": "REAL" | "FAKE" | "UNCERTAIN",
                "confidence": float (0.0 to 1.0),
                "explanation": str (human-readable reasoning),
                "metric_value": float (optional raw metric),
                "latency_ms": float (optional timing)
            }
        """
        pass

    def release(self):
        """Optional cleanup logic on shutdown."""
        pass
