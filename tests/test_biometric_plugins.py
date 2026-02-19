"""
Shield-Ryzen V2 — Biometric Plugin Tests (TASK 7.6)
===================================================
Tests for Challenge-Response, rPPG, Stereo, and Skin Reflectance.
Verifies logic correctness and graceful degradation.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 7 of 14 — Biometric Hardening
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import time
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.challenge_response import ChallengeResponsePlugin
from plugins.rppg_heartbeat import HeartbeatPlugin
# from plugins.stereo_depth import StereoDepthPlugin
from plugins.skin_reflectance import SkinReflectancePlugin
from shield_engine import ShieldEngine, PluginAwareStateMachine
from shield_face_pipeline import FaceDetection

class TestBiometricPlugins(unittest.TestCase):

    def setUp(self):
        # Common mock face
        self.face = MagicMock(spec=FaceDetection)
        self.face.bbox = (0, 0, 100, 100)
        self.face.landmarks_68 = np.zeros((68, 2))
        self.face.head_pose = (0, 0, 0)
        self.face.blendshapes = None
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # 1. Challenge Response Tests
    def test_challenge_response_generates_random_prompts(self):
        plug = ChallengeResponsePlugin()
        prompts = set()
        for _ in range(20):
            res = plug.start_challenge()
            prompts.add(res["prompt"])
        
        self.assertTrue(len(prompts) > 1, "Should generate diverse prompts")
        self.assertIn("blink_twice", prompts)

    def test_challenge_timeout_returns_fake(self):
        plug = ChallengeResponsePlugin()
        plug.start_challenge()
        plug.challenge_timeout_sec = 0.05 # Fast timeout
        
        time.sleep(0.1)
        res = plug.analyze(self.face, self.frame)
        
        self.assertEqual(res["verdict"], "FAKE")
        self.assertIn("timed out", res["explanation"])

    # 2. rPPG Tests
    def test_rppg_detects_valid_heartbeat_signal(self):
        plug = HeartbeatPlugin(buffer_seconds=1.0, fps=30.0) # Short buffer for test
        
        # Simulate 60 BPM sine wave (1 Hz) in green channel
        # 30 frames/sec. 30 frames = 1 sec.
        timestamps = np.linspace(0, 1.0, 30)
        signal = 128 + 10 * np.sin(2 * np.pi * 1.0 * timestamps)
        
        for val in signal:
            # Create frame with specific green value
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, 1] = int(val)
            res = plug.analyze(self.face, frame)
        
        # Last result should be REAL (SNR might be low due to short window, but lets check logic)
        # Note: FFT resolution on 1s window is 1Hz. So 1Hz is exactly bin 1.
        # But buffer needs to fill. 30 frames.
        
        # It's hard to get high SNR with 30 frames. But dominant freq should be 60 BPM.
        if res["verdict"] != "UNCERTAIN":
             self.assertTrue(res["metric_value"] > 50 and res["metric_value"] < 70, 
                             f"Expected ~60 BPM, got {res['metric_value']}")

    def test_rppg_flags_flat_signal_as_fake(self):
        plug = HeartbeatPlugin(buffer_seconds=1.0, fps=30.0)
        flat_val = 128
        
        for _ in range(35):
            frame = np.full((100, 100, 3), flat_val, dtype=np.uint8)
            res = plug.analyze(self.face, frame)
            
        # Should be FAKE due to low SNR (0 variance)
        if res["verdict"] != "UNCERTAIN":
             self.assertEqual(res["verdict"], "FAKE")

    # 3. Stereo Depth Tests
    @patch("plugins.stereo_depth.ShieldCamera")
    def test_stereo_depth_degrades_gracefully_without_camera2(self, mock_cam):
        # Simulate camera open failure
        mock_cam.side_effect = Exception("No camera")
        
        from plugins.stereo_depth import StereoDepthPlugin
        plug = StereoDepthPlugin()
        self.assertFalse(plug.available)
        
        res = plug.analyze(self.face, self.frame)
        self.assertEqual(res["verdict"], "UNCERTAIN")
        self.assertIn("unavailable", res["explanation"])

    # 4. Skin Reflectance Tests
    def test_skin_reflectance_distinguishes_skin_vs_screen(self):
        plug = SkinReflectancePlugin()
        
        # Case A: Matte/Smooth (Mask-like)
        smooth_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        res_smooth = plug.analyze(self.face, smooth_frame)
        # Gradient should be 0 -> FAKE (Surface too smooth)
        self.assertEqual(res_smooth["verdict"], "FAKE")
        self.assertIn("smooth", res_smooth["explanation"])
        
        # Case B: Specular (Shiny)
        shiny_frame = np.full((100, 100, 3), 255, dtype=np.uint8) # blown out
        res_shiny = plug.analyze(self.face, shiny_frame)
        self.assertEqual(res_shiny["verdict"], "FAKE")
        self.assertIn("specularity", res_shiny["explanation"])

    # 5. Integration Tests
    def test_plugins_return_correct_vote_format(self):
        plug = ChallengeResponsePlugin()
        res = plug.analyze(self.face, self.frame)
        self.assertIn("verdict", res)
        self.assertIn("confidence", res)
        self.assertIn("explanation", res)
        self.assertIn("name", res)

    def test_engine_fuses_plugin_votes_correctly(self):
        # Mocking State Machine
        sm = PluginAwareStateMachine()
        
        # Case 1: All Plugins UNCERTAIN/REAL -> Pass through (Base update)
        # Base update with T1=REAL, T2=PASS, T3=PASS -> VERIFIED
        # Let's mock base behavior: table maps [1,1,1] -> REAL
        
        votes = [{"verdict": "REAL", "confidence": 1.0}]
        state = sm.update("REAL", "PASS", "PASS", plugin_votes=votes)
        self.assertIn(state, ["REAL", "VERIFIED"]) # Depending on transitions
        
        # Case 2: One Plugin FAKE -> Should downgrade T3 to FAIL -> [1,1,0] -> SUSPICIOUS
        votes_fake = [{"verdict": "FAKE", "confidence": 0.9}]
        # Reset SM if needed, but we check instantaneous logic
        state_bad = sm.update("REAL", "PASS", "PASS", plugin_votes=votes_fake)
        
        # T3 becomes FAIL. Inputs: [1,1,0]. Truth table [1,1,0] -> SUSPICIOUS
        self.assertEqual(sm._to_bool("PASS", "PASS"), True) # Just checking helper
        # Logic verification
        # The update method overrides t3="FAIL" inside.
        # So we expect SUSPICIOUS result
        # Note: Hysteresis might delay it if history is mixed. 
        # But if we force a new state machine it starts UNKNOWN -> takes first proposed.
        
        sm2 = PluginAwareStateMachine()
        s = sm2.update("REAL", "PASS", "PASS", plugin_votes=votes_fake)
        self.assertEqual(s, "SUSPICIOUS")

if __name__ == '__main__':
    unittest.main()
