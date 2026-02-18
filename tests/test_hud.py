
import os
import sys
import unittest
import numpy as np
import time
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_types import EngineResult, FaceResult
from shield_hud import ShieldHUD

class TestShieldHUD(unittest.TestCase):

    def setUp(self):
        self.hud = ShieldHUD(use_audio=False)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.base_result = EngineResult(
            frame=self.frame,
            state="NO_FACE",
            face_results=[],
            fps=30.0,
            timing_breakdown={},
            camera_health={}
        )

    def test_render_does_not_crash_on_empty_results(self):
        annotated, t_hud = self.hud.render(self.frame, self.base_result)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertGreaterEqual(t_hud, 0)
        
        # Should contain "SYSTEM: NO_FACE" text
        # Difficult to verify pixel content, but execution success is key

    def test_render_handles_multi_face(self):
        # Create 2 mock faces
        f1 = FaceResult((10,10,50,50), "REAL", 0.9, 0.3, "HIGH", 0.0, "OK", ("REAL", "PASS", "PASS"), 0.0)
        f2 = FaceResult((100,10,50,50), "FAKE", 0.8, 0.3, "HIGH", 0.9, "BAD", ("FAKE", "PASS", "FAIL"), 0.0)
        
        self.base_result.face_results = [f1, f2]
        self.base_result.state = "REAL" # Overall state
        
        annotated, t_hud = self.hud.render(self.frame, self.base_result)
        self.assertIsNotNone(annotated)

    def test_all_states_have_distinct_shapes(self):
        # Verify that STATE_MAPPING covers necessary keys and COLORS has shapes
        # ShieldEngine uses: REAL, FAKE
        mapped_real = self.hud._get_mapped_state("REAL")
        mapped_fake = self.hud._get_mapped_state("FAKE")
        
        self.assertIn(mapped_real, self.hud.COLORS)
        self.assertIn(mapped_fake, self.hud.COLORS)
        
        shape_real = self.hud.COLORS[mapped_real]["shape"]
        shape_fake = self.hud.COLORS[mapped_fake]["shape"]
        
        self.assertNotEqual(shape_real, shape_fake)
        self.assertEqual(shape_real, "checkmark")
        self.assertEqual(shape_fake, "x_mark")

    def test_hud_render_time_under_5ms(self):
        # Benchmark simple render
        f1 = FaceResult((10,10,100,100), "REAL", 0.9, 0.3, "HIGH", 0.0, "OK", ("REAL", "PASS", "PASS"), 0.0)
        self.base_result.face_results = [f1]
        self.base_result.state = "REAL"
        
        times = []
        for _ in range(100):
            _, t = self.hud.render(self.frame, self.base_result)
            times.append(t)
            
        avg_time = sum(times) / len(times)
        # 5ms = 0.005s. Python CV2 drawing is very fast (usually <1ms)
        self.assertLess(avg_time, 0.005)

    def test_text_contrast_ratio_meets_wcag(self):
        # This is a static code check primarily, but we can verify logic is drawing outline
        # We check that for every text draw, there's a corresponding background/outline draw
        
        # We'll patch cv2.putText to verify calls
        with patch("cv2.putText") as mock_text:
            f1 = FaceResult((10,10,50,50), "REAL", 0.9, 0.3, "HIGH", 0.0, "OK", ("REAL", "PASS", "PASS"), 0.0)
            self.base_result.face_results = [f1]
            
            self.hud.render(self.frame, self.base_result)
            
            # Expect calls. For face text "VERIFIED 90%":
            # 1. Black Outline (thickness > 1)
            # 2. White Text (thickness 1)
            
            # Filter calls for "VERIFIED"
            calls = mock_text.call_args_list
            verified_calls = [c for c in calls if "VERIFIED" in c[0][1]]
            
            self.assertGreaterEqual(len(verified_calls), 2)
            
            # Check colors
            # Args: img, text, pos, font, scale, color, thickness
            outline_call = verified_calls[0]
            text_call = verified_calls[1]
            
            outline_color = outline_call[0][5]
            text_color = text_call[0][5]
            
            self.assertEqual(outline_color, (0,0,0)) # Black
            self.assertEqual(text_color, (255,255,255)) # White

if __name__ == "__main__":
    unittest.main()
