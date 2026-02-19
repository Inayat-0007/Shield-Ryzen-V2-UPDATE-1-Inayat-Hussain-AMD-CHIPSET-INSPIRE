"""
Shield-Ryzen V2 — Calibrated Decision Engine
===============================================
Re-exports DecisionStateMachine from shield_utils_core.py
and provides TierResult for backward compatibility.

States: VERIFIED, SUSPICIOUS, HIGH_RISK, UNKNOWN, REAL,
        WAIT_BLINK, FAKE, CRITICAL

Developer: Inayat Hussain | AMD Slingshot 2026
Part 3 of 14 — Liveness, Forensics & Decision Logic Calibration
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

# Ensure project root is importable
_project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shield_utils_core import DecisionStateMachine  # noqa: E402


# ===================================================================
# Tier Result Type (used by tests and evaluation scripts)
# ===================================================================

@dataclass
class TierResult:
    """Result from a single analysis tier.

    The DecisionStateMachine.update() accepts both raw strings
    ("REAL"/"FAKE", "PASS"/"FAIL") and TierResult objects.
    When a TierResult is passed, its `passed` field is used.
    """
    passed: bool               # True = real/pass, False = fake/fail
    confidence: float = 1.0    # 0.0 - 1.0
    detail: str = ""           # Human-readable explanation
    reliability: str = "HIGH"  # HIGH, MEDIUM, LOW


__all__ = [
    "DecisionStateMachine",
    "TierResult",
]
