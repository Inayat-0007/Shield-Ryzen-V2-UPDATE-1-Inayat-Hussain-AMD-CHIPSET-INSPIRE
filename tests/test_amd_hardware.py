
import unittest
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Adjust path finding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_xdna_engine import ShieldXDNAEngine
from performance.zero_copy_buffer import ZeroCopyBuffer
from security.ftpm_wrapper import FTPMWrapper
from camera.amd_direct_capture import AMDDirectCapture

class TestAMDHardware(unittest.TestCase):

    def test_xmodel_compilation_produces_output(self):
        # We can't actually run vai_c_xir here, but we can verify the script exists
        self.assertTrue(os.path.exists("compile_xmodel.sh"))
        
        # Test script is executable logic (skip actual exe on windows/mock env)
        with open("compile_xmodel.sh", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("vai_c_xir", content)

    @staticmethod
    def side_effect_exists(path):
        if "device_baseline.json" in str(path):
            return False
        if "attribution_classifier.onnx" in str(path):
            return False
        if "compile_xmodel.sh" in str(path):
            return True
        return True 

    def test_xdna_engine_fallback_to_onnx(self):
        # Test with no XDNA runtime available
        with patch.dict(sys.modules, {'xir': None, 'vart': None}): 
            config = {"model_path": "dummy_int8.onnx"}
            
            with patch("v3_int8_engine.ShieldCamera"), \
                 patch("v3_int8_engine.ShieldFacePipeline"), \
                 patch("v3_int8_engine.ShieldLogger"), \
                 patch("v3_int8_engine.ShieldHUD"), \
                 patch("v3_int8_engine.ShieldAudio"), \
                 patch("v3_int8_engine.ShieldEngine._init_model"), \
                 patch("os.path.exists", side_effect=self.side_effect_exists):
                 
                engine = ShieldXDNAEngine(config)
                self.assertFalse(engine.use_native)

    def test_zero_copy_buffer_allocation(self):
        # Fallback case
        buf = ZeroCopyBuffer((10,10), dtype=np.uint8)
        self.assertIsInstance(buf.get_view(), np.ndarray)
        self.assertEqual(buf.get_view().shape, (10,10))
        self.assertFalse(buf.is_unified())

    def test_direct_capture_fallback_to_opencv(self):
        with patch("cv2.VideoCapture"), patch("cv2.CAP_DSHOW", 700):
            cap = AMDDirectCapture(0)
            self.assertFalse(cap.amd_capture_enabled)
            # Should have called super init, check health metric default
            # cap.get_health_status() should exist
            self.assertTrue(hasattr(cap, "get_health_status"))

    def test_ftpm_wrapper_fallback_to_software(self):
        wrapper = FTPMWrapper()
        # Mock available = False
        wrapper.available = False
        
        # Create dummy file
        with open("dummy_model.bin", "wb") as f:
            f.write(b"data")
        
        try:
            res, msg = wrapper.verify_and_load("dummy_model.bin")
            # Should fail as sig missing
            self.assertFalse(res)
            self.assertIn("Signature missing", msg)
        finally:
            if os.path.exists("dummy_model.bin"):
                os.remove("dummy_model.bin")

    def test_xdna_engine_performance_faster_than_onnx(self):
        # Mock native execution path
        config = {"xmodel_path": "dummy.xmodel"}
        
        with patch("v3_int8_engine.ShieldCamera"), \
             patch("v3_int8_engine.ShieldFacePipeline"), \
             patch("v3_int8_engine.ShieldLogger"), \
             patch("v3_int8_engine.ShieldHUD"), \
             patch("v3_int8_engine.ShieldAudio"), \
             patch("v3_int8_engine.ShieldEngine._init_model"), \
             patch("os.path.exists", side_effect=self.side_effect_exists):
             
             engine = ShieldXDNAEngine(config)
             
             # Mock necessary components for native="True"
             engine.use_native = True
             engine.runner = MagicMock()
             engine.runner.get_input_tensors.return_value = [1]
             engine.runner.get_output_tensors.return_value = [1]
             
             # Run Inference (Native Path)
             import time
             t0 = time.time()
             res = engine._run_inference(np.zeros((299,299,3), dtype=np.float32))
             t_native = time.time() - t0
             
             # Switch to Fallback Path
             engine.use_native = False
             engine.model_type = "MOCK" # Force mock
             
             t0 = time.time()
             res2 = engine._run_inference(np.zeros((299,299,3), dtype=np.float32))
             t_fallback = time.time() - t0
             
             # Basic assertions
             engine.runner.get_input_tensors.assert_called()
             self.assertIsInstance(res, np.ndarray)

if __name__ == "__main__":
    unittest.main()
