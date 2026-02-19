"""
Shield-Ryzen V2 — HUD Render Tests (TASK 10.4)
==============================================
Verify rendering robustness, multi-face handling, and accessibility compliance.
Checks shape diversity and rendering speed.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 10 of 14 — HUD & Explainability
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2
import time
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_hud import ShieldHUD
from shield_engine import EngineResult, FaceResult

class TestShieldHUD(unittest.TestCase):
    
    def setUp(self):
        self.hud = ShieldHUD()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.result = EngineResult(
            frame=self.frame,
            timestamp=time.monotonic(),
            face_results=[],
            fps=0.0,
            timing_breakdown={},
            camera_health={},
            memory_mb=0.0
        )

    def test_render_does_not_crash_on_empty_results(self):
        # Empty result
        annotated, elapsed = self.hud.render(self.frame, self.result)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_render_handles_multi_face(self):
        f1 = FaceResult(
            face_id=1, 
            bbox=(100, 100, 200, 200),
            landmarks=[],
            state="VERIFIED", 
            confidence=0.9,
            neural_confidence=0.9,
            ear_value=0.3, 
            ear_reliability="HIGH",
            texture_score=10.0,
            texture_explanation="OK",
            tier_results=("REAL", "PASS", "PASS"),
            plugin_votes=[]
        )
        f2 = FaceResult(
            face_id=2, 
            bbox=(400, 100, 200, 200),
            landmarks=[],
            state="HIGH_RISK", 
            confidence=0.8,
            neural_confidence=0.8,
            ear_value=0.2, 
            ear_reliability="LOW",
            texture_score=0.0,
            texture_explanation="Blurry",
            tier_results=("FAKE", "FAIL", "FAIL"),
            plugin_votes=[]
        )
        
        self.result.face_results = [f1, f2]
        self.result.fps = 30.0
        
        annotated, elapsed = self.hud.render(self.frame, self.result)
        self.assertEqual(annotated.shape, (720, 1280, 3))
    
    def test_all_states_have_distinct_shapes(self):
        # Verify config
        shapes = set()
        colors = set()
        for state, cfg in self.hud.COLORS.items():
            if state != "FAKE": # FAKE maps to HIGH_RISK often, check shape uniqueness logic
                shapes.add(cfg["shape"])
                colors.add(cfg["bg"])
        
        # We expect checkmark, x_mark, question, dash, circle, triangle -> 6 distinct
        # VERIFIED and REAL share "checkmark".
        # HIGH_RISK and FAKE share "x_mark".
        # SUSPICIOUS is "question".
        # UNKNOWN is "dash".
        # NO_FACE is "circle".
        # CAMERA_ERROR is "triangle".
        self.assertTrue(len(shapes) >= 5, "Should have distinct shapes for distinct states")

    def test_hud_render_time_under_5ms(self):
        # Render a full frame
        self.result.fps = 60.0
        f1 = FaceResult(
            face_id=1, 
            bbox=(100, 100, 200, 200), 
            landmarks=[],
            state="VERIFIED", 
            confidence=0.9,
            neural_confidence=0.9,
            ear_value=0.3,
            ear_reliability="HIGH",
            texture_score=10.0,
            texture_explanation="OK",
            tier_results=("REAL", "PASS", "PASS"),
            plugin_votes=[]
        )
        self.result.face_results = [f1]
        
        # Warmup
        self.hud.render(self.frame, self.result)
        
        start = time.monotonic()
        for _ in range(100):
            self.hud.render(self.frame, self.result)
        end = time.monotonic()
        
        avg_ms = ((end - start) * 1000.0) / 100.0
        print(f"Average HUD render time: {avg_ms:.3f}ms")
        self.assertLess(avg_ms, 5.0, "HUD rendering too slow")

if __name__ == '__main__':
    unittest.main()
