
import cv2
import time
import logging

try:
    from shield_camera import ShieldCamera
except ImportError:
    # If using relative path or run from tests
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shield_camera import ShieldCamera

_log = logging.getLogger("AMDDirectCapture")

class AMDDirectCapture(ShieldCamera):
    """
    Extends ShieldCamera with AMD driver-level capture.
    Lower latency than OpenCV by bypassing DirectShow buffer.
    Falls back to ShieldCamera if AMD SDK unavailable.
    """
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        super().__init__(camera_id, backend=cv2.CAP_DSHOW) # Init base just in case
        self.amd_capture_enabled = False
        
        # Try importing AMD optimized capture library (Hypothetical)
        try:
            import amd_camera_sdk
            self.capture_handle = amd_camera_sdk.open_device(camera_id, width, height, fps)
            self.amd_capture_enabled = True
            _log.info("AMD Direct Capture initialized successfully")
        except ImportError:
            _log.warning("AMD Camera SDK not found. Using Standard OpenCV capture.")
            self.amd_capture_enabled = False

    def read_validated_frame(self):
        """Override standard read to use AMD capture if available."""
        if self.amd_capture_enabled:
            import amd_camera_sdk
            ret, frame, ts = amd_camera_sdk.read_frame(self.capture_handle)
            if not ret:
                self._update_health_metrics(False)
                return False, None, 0.0
            
            self._update_health_metrics(True)
            return True, frame, ts
        else:
            # Fallback to base class implementation
            return super().read_validated_frame()

    def release(self):
        if self.amd_capture_enabled:
            import amd_camera_sdk
            amd_camera_sdk.close_device(self.capture_handle)
        else:
            super().release()
