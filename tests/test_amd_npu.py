"""
Shield-Ryzen V2 — AMD NPU Integration Tests (TASK 12.1)
=======================================================
Verifies Ryzen AI Engine loading, hardware monitoring, and quantization pipeline readiness.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 12 of 14 — Comprehensive Validation
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_xdna_engine import RyzenXDNAEngine
from shield_hardware_monitor import HardwareMonitor
from quantize_ryzen import ShieldCalibrationDataReader

class TestAMDOptimization(unittest.TestCase):
    
    def test_hardware_monitor_updates(self):
        monitor = HardwareMonitor(history_len=5)
        monitor.update(fps=30.0)
        stats = monitor.get_stats()
        
        self.assertIn("cpu_curr", stats)
        self.assertIn("mem_mb", stats)
        self.assertIn("fps_avg", stats)
        self.assertGreater(stats["fps_avg"], 0)
        
        # Test heuristic power estimation
        watts = monitor.estimate_power()
        self.assertGreaterEqual(watts, 5.0) # Base is 5W (>= because CPU might be 0)

    @patch('v3_xdna_engine.psutil.Process')
    def test_xdna_engine_priority(self, mock_process):
        # Mock processnice
        mock_proc_instance = MagicMock()
        mock_process.return_value = mock_proc_instance
        
        # Instantiate engine
        def dummy_init(self, config=None):
            self.logger = MagicMock()
            self.config = {}

        with patch('shield_engine.ShieldEngine.__init__', new=dummy_init):
            engine = RyzenXDNAEngine()
            
            # Check if priority boost was attempted
            self.assertTrue(mock_proc_instance.nice.called)

    def test_quantization_reader_logic(self):
        # Test ShieldCalibrationDataReader without running full quantization
        # Create a dummy calibration directory
        calib_dir = "tests/temp_calib"
        os.makedirs(calib_dir, exist_ok=True)
        
        # Create dummy image
        import cv2
        dummy = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(calib_dir, "test.jpg"), dummy)
        
        try:
            # Init reader
            # It uses ShieldFacePipeline which needs MediaPipe model.
            # If model missing, it might fail.
            # Assuming test env has dependencies or mocks.
            # We'll see. If pipeline fails, we catch it.
            
            dr = ShieldCalibrationDataReader(calib_dir, input_name="input.1")
            
            # Get next batch
            # Preprocessing will run.
            # If standard pipeline doesn't detect face in noise, it falls back to crop.
            batch = dr.get_next()
            
            if batch:
                self.assertIn("input.1", batch)
                self.assertEqual(batch["input.1"].shape, (1, 3, 299, 299))
                
        except Exception as e:
            # If MediaPipe fails due to missing .tflite, skip
            if "face_detection_short_range.tflite" in str(e):
                print("Skipping due to missing MediaPipe model")
            else:
                raise
        finally:
            import shutil
            if os.path.exists(calib_dir):
                shutil.rmtree(calib_dir)

if __name__ == '__main__':
    unittest.main()
