"""
Shield-Ryzen V2 — Engine Integration Tests (TASK 6.6)
=====================================================
Validates the unified ShieldEngine, including:
- Triple-buffer async architecture (simulated)
- Plugin system
- Logging
- State machine integration

Developer: Inayat Hussain | AMD Slingshot 2026
Part 6 of 14 — Integration & Efficiency
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_engine import ShieldEngine, EngineResult, FaceResult
from shield_plugin import ShieldPlugin

class MockPlugin(ShieldPlugin):
    @property
    def name(self): return "MockPlugin"
    @property
    def tier(self): return "test"
    def analyze(self, face, frame):
        return {
            "verdict": "REAL",
            "confidence": 0.9,
            "name": self.name
        }

class TestShieldEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a dummy config
        cls.config = {
            "camera_id": 0,
            "detector_type": "mediapipe",
            "model_path": "shield_ryzen_v2.onnx", # Tests assume generic path, will be mocked
            "log_path": "logs/test_audit.jsonl"
        }

    def setUp(self):
        # Patch dependencies before init
        self.cam_patcher = patch('shield_engine.ShieldCamera')
        self.pipe_patcher = patch('shield_engine.ShieldFacePipeline')
        self.xcp_patcher = patch('shield_engine.ShieldXception') # For PyTorch path
        self.ort_patcher = patch('onnxruntime.InferenceSession') # For ONNX path
        
        self.mock_cam_cls = self.cam_patcher.start()
        self.mock_pipe_cls = self.pipe_patcher.start()
        self.mock_xcp_cls = self.xcp_patcher.start()
        self.mock_ort_cls = self.ort_patcher.start()
        
        # Setup mocks
        self.mock_cam = self.mock_cam_cls.return_value
        self.mock_cam.read_validated_frame.return_value = (True, np.zeros((480,640,3), dtype=np.uint8), time.time())
        self.mock_cam.get_health_status.return_value = {"status": "OK"}
        
        self.mock_pipe = self.mock_pipe_cls.return_value
        self.mock_pipe.detect_faces.return_value = []
        
        # Mock ONNX session
        self.mock_session = self.mock_ort_cls.return_value
        # Use simple object that is serializable
        mock_input = MagicMock()
        mock_input.name = "input_1"
        self.mock_session.get_inputs.return_value = [mock_input]
        self.mock_session.get_providers.return_value = ["CPUExecutionProvider"] # Serializable list
        self.mock_session.run.return_value = [[np.array([0.1, 0.9])]] # Fake=0.1, Real=0.9
        
    def tearDown(self):
        self.cam_patcher.stop()
        self.pipe_patcher.stop()
        self.xcp_patcher.stop()
        self.ort_patcher.stop()

    def test_engine_initializes_all_modules(self):
        engine = ShieldEngine(self.config)
        self.mock_cam_cls.assert_called()
        self.mock_pipe_cls.assert_called()
        # Verify queues
        self.assertIsNotNone(engine.camera_queue)
        self.assertIsNotNone(engine.result_queue)

    def test_process_frame_returns_engine_result(self):
        engine = ShieldEngine(self.config)
        # Mock face detection
        mock_face = MagicMock()
        mock_face.bbox = (100, 100, 200, 200)
        mock_face.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32)
        mock_face.face_crop_raw = np.zeros((200,200,3), dtype=np.uint8)
        mock_face.landmarks_68 = np.zeros((68,2))
        mock_face.head_pose = (0,0,0)
        mock_face.is_frontal = True
        mock_face.blendshapes = None
        
        self.mock_pipe.detect_faces.return_value = [mock_face]
        
        frame = np.zeros((480,640,3), dtype=np.uint8)
        result = engine._process_frame(frame, time.time())
        
        self.assertIsInstance(result, EngineResult)
        self.assertEqual(len(result.face_results), 1)
        # We accept HIGH_RISK or WAIT_BLINK because our mock image (zeros) fails texture check
        # and has 0 blinks. The important thing is it ran through the logic.
        self.assertIn(result.face_results[0].state, ["REAL", "WAIT_BLINK", "HIGH_RISK"])
        # Verify neural confidence propagated
        self.assertGreater(result.face_results[0].neural_confidence, 0.5)

    def test_no_face_returns_no_results(self):
        engine = ShieldEngine(self.config)
        self.mock_pipe.detect_faces.return_value = []
        frame = np.zeros((480,640,3), dtype=np.uint8)
        result = engine._process_frame(frame, time.time())
        self.assertEqual(len(result.face_results), 0)

    def test_multi_face_returns_multiple_results(self):
        engine = ShieldEngine(self.config)
        f1 = MagicMock(); f1.bbox = (0,0,10,10); f1.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32); f1.face_crop_raw = np.zeros((10,10,3), dtype=np.uint8)
        f2 = MagicMock(); f2.bbox = (20,20,10,10); f2.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32); f2.face_crop_raw = np.zeros((10,10,3), dtype=np.uint8)
        # minimal attributes
        for f in [f1, f2]:
            f.landmarks_68 = np.zeros((68,2))
            f.head_pose = (0,0,0)
            f.is_frontal = True
            f.blendshapes = None
        
        self.mock_pipe.detect_faces.return_value = [f1, f2]
        result = engine._process_frame(np.zeros((480,640,3), dtype=np.uint8), time.time())
        self.assertEqual(len(result.face_results), 2)

    def test_fps_measurement_is_end_to_end(self):
        engine = ShieldEngine(self.config)
        # process a few frames
        for _ in range(5):
             engine._process_frame(np.zeros((100,100,3), dtype=np.uint8), time.time())
        
        # Check internal fps tracker
        self.assertTrue(len(engine._frame_times) > 0)
        # Calculate fps
        fps = len(engine._frame_times) / sum(engine._frame_times)
        self.assertTrue(fps > 0)

    def test_logging_produces_valid_jsonl(self):
        # We check if logger was called
        engine = ShieldEngine(self.config)
        with patch.object(engine.logger, 'log_frame') as mock_log:
             engine._process_frame(np.zeros((100,100,3), dtype=np.uint8), time.time())
             mock_log.assert_called_once()
             args = mock_log.call_args[0][0]
             self.assertIn("timestamp", args)
             self.assertIn("fps", args)

    def test_plugin_registration_and_voting(self):
        engine = ShieldEngine(self.config)
        p = MockPlugin()
        engine.register_plugin(p)
        self.assertEqual(len(engine.plugins), 1)
        
        # Test if plugin is called during process
        f1 = MagicMock()
        f1.bbox = (0,0,10,10)
        f1.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32)
        f1.face_crop_raw = np.zeros((10,10,3), dtype=np.uint8)
        f1.landmarks_68 = np.zeros((68,2))
        f1.head_pose = (0,0,0)
        f1.is_frontal = True
        f1.blendshapes = None
        self.mock_pipe.detect_faces.return_value = [f1]
        
        with patch.object(p, 'analyze', wraps=p.analyze) as mock_analyze:
             res = engine._process_frame(np.zeros((100,100,3), dtype=np.uint8), time.time())
             mock_analyze.assert_called()
             # Check result contains plugin vote
             self.assertEqual(len(res.face_results[0].plugin_votes), 1)
             self.assertEqual(res.face_results[0].plugin_votes[0]["name"], "MockPlugin")

if __name__ == '__main__':
    unittest.main()
