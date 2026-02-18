
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

# Adjust path to import engine
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_int8_engine import ShieldEngine, EngineResult
from shield_face_pipeline import FaceDetection

class TestShieldEngine(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            "camera_id": 0,
            "detector_type": "mediapipe",
            "model_path": "shield_ryzen_int8.onnx", 
            "log_path": "logs/test_audit.jsonl"
        }
        # Common patcher for open to avoid repeating
        self.open_patcher = patch("builtins.open", mock_open(read_data='{"baseline_texture": 0.5}'))
        self.mock_open = self.open_patcher.start()

    def tearDown(self):
        self.open_patcher.stop()

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    @patch("os.path.exists", return_value=True) 
    @patch("v3_int8_engine.ConfidenceCalibrator")
    def test_engine_initializes_all_modules(self, MockCalib, MockExists, MockLogger, MockPipeline, MockCamera):
        # Mock ONNX session loading to avoid real file need
        with patch("onnxruntime.InferenceSession") as MockSession:
            engine = ShieldEngine(self.config)
            
            MockCamera.assert_called_once()
            MockPipeline.assert_called_once()
            MockLogger.assert_called_once()
            MockCalib.assert_called_once()
            self.assertTrue(MockSession.call_count >= 1) # Main model + optional modules
            # Check main model loaded
            # Note: arguments might be passed differently (providers etc), so we just check call count or fuzzy args
            # actual call is InferenceSession(model_path, providers=...)
            # We can check if any call arg[0] == model_path
            found_main_model = False
            for call in MockSession.call_args_list:
                args, _ = call
                if args and "shield_ryzen_int8.onnx" in args[0]:
                    found_main_model = True
                    break
            self.assertTrue(found_main_model, "Main ONNX model not loaded")
            
            self.assertIsNotNone(engine.camera)
            self.assertIsNotNone(engine.face_pipeline)

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    @patch("os.path.exists", return_value=True)
    def test_process_frame_returns_engine_result(self, MockExists, MockLogger, MockPipeline, MockCamera):
        # Setup mocks
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8), time.monotonic())
        mock_cam.check_frame_freshness.return_value = True
        # Fix: Provide dict with float/int values for HUD formatting
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        mock_pipe = MockPipeline.return_value
        mock_pipe.detect_faces.return_value = [] # No face case first
        
        with patch("onnxruntime.InferenceSession"):
            engine = ShieldEngine(self.config)
            result = engine.process_frame()
            
            self.assertIsInstance(result, EngineResult)
            self.assertEqual(result.state, "NO_FACE") # With HUD it renders NO_FACE state correctly
            self.assertIsInstance(result.timing_breakdown, dict)

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    def test_no_face_with_timing(self, MockLogger, MockPipeline, MockCamera):
        # Similar to above but check timing keys
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((100,100,3), dtype=np.uint8), 1.0)
        mock_cam.check_frame_freshness.return_value = True
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        mock_pipe = MockPipeline.return_value
        mock_pipe.detect_faces.return_value = []
        
        with patch("onnxruntime.InferenceSession"), patch("os.path.exists", return_value=True):
            engine = ShieldEngine(self.config)
            result = engine.process_frame()
            
            self.assertIn("capture_ms", result.timing_breakdown)
            self.assertIn("detect_ms", result.timing_breakdown)
            # Infer/Liveness/Texture keys might be missing if logic skips them on no_face?
            # Implementation puts them inside "if not faces" block?
            # Let's check implementation. 
            # Implementation returns EARLY if not faces.
            # So breakdown only has capture/detect.
            # This is correct valid behavior.

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    @patch("v3_int8_engine.compute_ear", return_value=(0.3, "HIGH"))
    @patch("v3_int8_engine.compute_texture_score", return_value=(0.9, False, "Good"))
    @patch("v3_int8_engine.ConfidenceCalibrator")
    def test_multi_face_returns_multiple_results(self, MockCalib, MockTexture, MockEar, MockLogger, MockPipeline, MockCamera):
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((100,100,3), dtype=np.uint8), 1.0)
        mock_cam.check_frame_freshness.return_value = True
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        # Mock 2 faces
        face1 = MagicMock(spec=FaceDetection)
        face1.bbox = (0,0,50,50)
        face1.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32)
        face1.face_crop_raw = np.zeros((50,50,3), dtype=np.uint8)
        face1.landmarks = np.zeros((478,2))
        
        face2 = MagicMock(spec=FaceDetection)
        face2.bbox = (60,60,40,40)
        face2.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32)
        face2.face_crop_raw = np.zeros((40,40,3), dtype=np.uint8)
        face2.landmarks = np.zeros((478,2))
        
        mock_pipe = MockPipeline.return_value
        mock_pipe.detect_faces.return_value = [face1, face2]
        
        with patch("onnxruntime.InferenceSession") as MockSession, \
             patch("os.path.exists", return_value=True):
             
            # Mock Inference Session run return
            session_instance = MockSession.return_value
            # configure inputs/outputs
            session_instance.get_inputs.return_value = [MagicMock(name="input")]
            session_instance.get_providers.return_value = ["CPU"]
            # run returns list of outputs. Output 0 is (1,2) probs array.
            session_instance.run.return_value = [np.array([[0.1, 0.9]], dtype=np.float32)]
            
            engine = ShieldEngine(self.config)
            # Fix: Return Low Fake Probability (0.1) so result is REAL
            # calibrated = [ProbReal, ProbFake]
            MockCalib.return_value.calibrate.return_value = [0.9, 0.1]
            
            result = engine.process_frame()
            
            self.assertEqual(len(result.face_results), 2)
            self.assertEqual(result.face_results[0].state, "REAL") 
            
            # Re-run process with High Fake Prob
            MockCalib.return_value.calibrate.return_value = [0.1, 0.9]
            result = engine.process_frame()
            # Hysteresis prevents immediate flip unless consistent?
            # StateMachine needs continuous updates.
            # But here we just check if value propagation works.
            # StateMachine update logic: if Tier1=FAKE -> update(FAKE).
            # If history was empty, FAKE.
            # If we run once, history has 1 FAKE. result is FAKE.
            self.assertEqual(result.face_results[0].state, "FAKE")

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    def test_fps_measurement_is_end_to_end(self, MockLogger, MockPipeline, MockCamera):
        # Run multiple frames and check FPS value
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((100,100,3), dtype=np.uint8), 1.0)
        mock_cam.check_frame_freshness.return_value = True
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        mock_pipe = MockPipeline.return_value
        mock_pipe.detect_faces.return_value = []
        
        with patch("onnxruntime.InferenceSession"), patch("os.path.exists", return_value=True):
            engine = ShieldEngine(self.config)
            # engine._frame_times is empty
            r1 = engine.process_frame()
            self.assertGreater(r1.fps, 0)
            
            r2 = engine.process_frame()
            self.assertIsInstance(r2.fps, float)

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    @patch("psutil.Process")
    @patch("gc.collect")
    def test_memory_growth_triggers_gc(self, MockGC, MockProcess, MockLogger, MockPipeline, MockCamera):
        # Setup memory values
        proc = MockProcess.return_value
        # Baseline = 100MB
        # Current = 700MB (>500MB growth)
        info1 = MagicMock(rss=100 * 1024 * 1024)
        info2 = MagicMock(rss=700 * 1024 * 1024)
        
        # We need side_effect for memory_info logic.
        # init calls it once (baseline).
        # process_frame calls it once (current).
        # We also need read_validated_frame and detect_faces to return something.
        
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((100,100,3), dtype=np.uint8), 1.0)
        mock_cam.check_frame_freshness.return_value = True
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        # No faces to keep it simple, detection phase still covered
        MockPipeline.return_value.detect_faces.return_value = []
        
        proc.memory_info.side_effect = [info1, info2, info2] # Init, Proccess call (Trigger), Reset?
        
        with patch("onnxruntime.InferenceSession"), patch("os.path.exists", return_value=True):
            engine = ShieldEngine(self.config) # Base 100
            engine.process_frame() # Current 700 -> Diff 600 > 500
            
            MockGC.assert_called_once()
            MockLogger.return_value.warn.assert_called()

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    def test_logging_produces_valid_jsonl(self, MockLogger, MockPipeline, MockCamera):
        # We can mock ShieldLogger instance log_frame call check
        instance = MockLogger.return_value
        
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((100,100,3), dtype=np.uint8), 1.0)
        mock_cam.check_frame_freshness.return_value = True
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        MockPipeline.return_value.detect_faces.return_value = []
        
        with patch("onnxruntime.InferenceSession"), patch("os.path.exists", return_value=True):
            engine = ShieldEngine(self.config)
            engine.process_frame()
            
            instance.log_frame.assert_called()
            # Inspect args
            args, _ = instance.log_frame.call_args
            data = args[0]
            self.assertIn("timestamp", data)
            self.assertIn("fps", data)
            self.assertIn("timing", data)

    @patch("v3_int8_engine.ShieldCamera")
    @patch("v3_int8_engine.ShieldFacePipeline")
    @patch("v3_int8_engine.ShieldLogger")
    @patch("v3_int8_engine.ConfidenceCalibrator")
    def test_timing_breakdown_all_stages_present(self, MockCalib, MockLogger, MockPipeline, MockCamera):
        # Use simple NO_FACE case again, as implementation returns early
        # If we want ALL stages, we need a face.
        mock_cam = MockCamera.return_value
        mock_cam.read_validated_frame.return_value = (True, np.zeros((100,100,3), dtype=np.uint8), 1.0)
        mock_cam.check_frame_freshness.return_value = True
        mock_cam.get_health_status.return_value = {"fps_actual": 30.0, "drop_rate_pct": 0.0}
        
        face = MagicMock(spec=FaceDetection)
        face.bbox = (0,0,1,1)
        face.face_crop_299 = np.zeros((1,3,299,299), dtype=np.float32)
        face.face_crop_raw = np.zeros((1,1,3), dtype=np.uint8)
        face.landmarks = np.zeros((478,2))
        
        MockPipeline.return_value.detect_faces.return_value = [face]
        
        with patch("onnxruntime.InferenceSession") as MockSession, \
             patch("os.path.exists", return_value=True), \
             patch("v3_int8_engine.compute_ear", return_value=(0.5, "HIGH")), \
             patch("v3_int8_engine.compute_texture_score", return_value=(0.1, False, "OK")):
             
            MockSession.return_value.run.return_value = [np.array([[0.5, 0.5]], dtype=np.float32)]
            
            engine = ShieldEngine(self.config)
            MockCalib.return_value.calibrate.return_value = [0.5, 0.5]
            
            result = engine.process_frame()
            
            # Check keys
            keys = result.timing_breakdown.keys()
            self.assertIn("capture_ms", keys)
            self.assertIn("detect_ms", keys)
            self.assertIn("infer_total_ms", keys)
            self.assertIn("liveness_total_ms", keys)
            self.assertIn("texture_total_ms", keys)
            self.assertIn("total_ms", keys)

if __name__ == "__main__":
    unittest.main()
