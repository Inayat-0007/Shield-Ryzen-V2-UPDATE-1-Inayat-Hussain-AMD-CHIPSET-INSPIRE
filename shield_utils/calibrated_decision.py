"""
Shield-Ryzen V2 -- Calibrated Decision Engine
===============================================
State machine with hysteresis for stable trust decisions.

Features:
  - Conflict resolution truth table (8 cases)
  - Temporal hysteresis (N consecutive frames required for state change)
  - Full decision history (300-frame rolling window)
  - Prevents flickering between VERIFIED/SUSPICIOUS states

States: VERIFIED, SUSPICIOUS, HIGH_RISK, UNKNOWN
Transitions require `hysteresis_frames` consecutive agreement.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 3 of 12 -- Liveness, Forensic Analysis & Calibration
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("DecisionEngine")


# ===================================================================
# Tier Result Types
# ===================================================================

@dataclass
class TierResult:
    """Result from a single analysis tier."""
    passed: bool               # True = real/pass, False = fake/fail
    confidence: float = 1.0    # 0.0 - 1.0
    detail: str = ""           # Human-readable explanation
    reliability: str = "HIGH"  # HIGH, MEDIUM, LOW


# ===================================================================
# Decision State Machine
# ===================================================================

class DecisionStateMachine:
    """Temporal state machine for stable security decisions.

    CONFLICT RESOLUTION TRUTH TABLE:

    Neural | Liveness | Forensic | -> Decision
    -------|----------|----------|----------
    Real   | Pass     | Pass     | -> VERIFIED
    Real   | Pass     | Fail     | -> SUSPICIOUS
    Real   | Fail     | Pass     | -> SUSPICIOUS
    Real   | Fail     | Fail     | -> HIGH_RISK
    Fake   | Pass     | Pass     | -> SUSPICIOUS
    Fake   | Pass     | Fail     | -> HIGH_RISK
    Fake   | Fail     | Pass     | -> HIGH_RISK
    Fake   | Fail     | Fail     | -> HIGH_RISK

    Rules:
    - ANY tier saying Fake/Fail prevents VERIFIED
    - Neural 'Fake' always escalates to SUSPICIOUS+
    - Two or more failures -> HIGH_RISK
    - State changes require N consecutive frames (hysteresis)
    """

    VALID_STATES = ("VERIFIED", "SUSPICIOUS", "HIGH_RISK", "UNKNOWN")

    def __init__(self, hysteresis_frames: int = 5):
        """Initialize decision state machine.

        Args:
            hysteresis_frames: Number of consecutive frames required
                               before a state transition is accepted.
                               Higher = more stable but slower response.
                               Default 5 = ~167ms at 30 FPS.
        """
        self.current_state: str = "UNKNOWN"
        self.pending_state: Optional[str] = None
        self.pending_count: int = 0
        self.hysteresis: int = hysteresis_frames
        self.history: deque = deque(maxlen=300)  # 10 sec at 30 FPS
        self._state_entry_time: float = time.monotonic()
        self._total_transitions: int = 0

    def update(
        self,
        tier1_result: TierResult,
        tier2_result: TierResult,
        tier3_result: TierResult,
    ) -> str:
        """Process one frame's tier results and return stable state.

        Args:
            tier1_result: Neural network classification result.
            tier2_result: Liveness (EAR/blink) check result.
            tier3_result: Forensic (texture/frequency) check result.

        Returns:
            Stable state after hysteresis: one of VALID_STATES.
        """
        proposed = self._resolve_conflict(tier1_result, tier2_result, tier3_result)

        # Hysteresis: require N consecutive frames proposing the same state
        if proposed == self.pending_state:
            self.pending_count += 1
        else:
            self.pending_state = proposed
            self.pending_count = 1

        if self.pending_count >= self.hysteresis:
            if self.current_state != proposed:
                _log.debug(
                    "State transition: %s -> %s (after %d frames)",
                    self.current_state, proposed, self.hysteresis,
                )
                self._total_transitions += 1
                self._state_entry_time = time.monotonic()
            self.current_state = proposed

        # Record to history
        self.history.append({
            "timestamp": time.monotonic(),
            "proposed": proposed,
            "stable": self.current_state,
            "tier_results": (
                tier1_result.passed,
                tier2_result.passed,
                tier3_result.passed,
            ),
            "pending_count": self.pending_count,
        })

        return self.current_state

    def get_state_duration_ms(self) -> float:
        """How long the current state has been active (ms)."""
        return (time.monotonic() - self._state_entry_time) * 1000.0

    def get_stability_score(self) -> float:
        """Score indicating decision stability (0.0=flickering, 1.0=stable).

        Based on the ratio of actual transitions vs potential transitions.
        """
        if len(self.history) < 10:
            return 0.5  # Not enough data

        recent = list(self.history)[-30:]
        proposed_states = [h["proposed"] for h in recent]
        unique_states = len(set(proposed_states))

        # 1 unique state = perfectly stable, 4 = maximum flickering
        return round(max(0.0, 1.0 - (unique_states - 1) / 3.0), 3)

    def reset(self) -> None:
        """Reset state machine to UNKNOWN."""
        self.current_state = "UNKNOWN"
        self.pending_state = None
        self.pending_count = 0
        self.history.clear()
        self._state_entry_time = time.monotonic()

    @staticmethod
    def _resolve_conflict(
        tier1: TierResult,
        tier2: TierResult,
        tier3: TierResult,
    ) -> str:
        """Apply the conflict resolution truth table.

        TRUTH TABLE:
          Neural=Real  + Liveness=Pass + Forensic=Pass -> VERIFIED
          Neural=Real  + Liveness=Pass + Forensic=Fail -> SUSPICIOUS
          Neural=Real  + Liveness=Fail + Forensic=Pass -> SUSPICIOUS
          Neural=Real  + Liveness=Fail + Forensic=Fail -> HIGH_RISK
          Neural=Fake  + Liveness=Pass + Forensic=Pass -> SUSPICIOUS
          Neural=Fake  + Liveness=Pass + Forensic=Fail -> HIGH_RISK
          Neural=Fake  + Liveness=Fail + Forensic=Pass -> HIGH_RISK
          Neural=Fake  + Liveness=Fail + Forensic=Fail -> HIGH_RISK
        """
        neural_real = tier1.passed
        liveness_pass = tier2.passed
        forensic_pass = tier3.passed

        failure_count = sum([
            not neural_real,
            not liveness_pass,
            not forensic_pass,
        ])

        if failure_count == 0:
            # All three pass -> VERIFIED
            return "VERIFIED"
        elif failure_count >= 2:
            # Two or more failures + always HIGH_RISK
            return "HIGH_RISK"
        else:
            # Exactly one failure
            if not neural_real:
                # Neural says Fake + other two pass -> SUSPICIOUS
                # (Neural Fake always escalates to at least SUSPICIOUS)
                return "SUSPICIOUS"
            else:
                # Neural Real but one other failed -> SUSPICIOUS
                return "SUSPICIOUS"

    def get_summary(self) -> dict:
        """Return current state machine summary."""
        return {
            "current_state": self.current_state,
            "pending_state": self.pending_state,
            "pending_count": self.pending_count,
            "hysteresis_threshold": self.hysteresis,
            "state_duration_ms": round(self.get_state_duration_ms(), 1),
            "stability_score": self.get_stability_score(),
            "total_transitions": self._total_transitions,
            "history_length": len(self.history),
        }
