
import threading
import time
import logging

_log = logging.getLogger("ShieldAudio")

class ShieldAudio:
    """Non-blocking audio alerts for system state transitions.
    
    Uses standard OS capabilities (Beep on Windows, TBD on Linux).
    Designed to be thread-safe and non-blocking.
    """

    def __init__(self, use_audio: bool = True):
        self.enabled = use_audio
        self._last_state = "UNKNOWN"
        self._lock = threading.Lock()
        
        # Determine beep function
        try:
            import winsound
            self._beep = winsound.Beep
            self._platform = "windows"
        except ImportError:
            self._beep = lambda freq, dur: None # Fallback
            self._platform = "other"
            _log.info("Audio disabled (platform specific library missing)")

    def update(self, current_state: str):
        """Check for state transitions and trigger audio alert if necessary.
        
        Args:
            current_state: The new system state ('REAL', 'FAKE', etc.)
        """
        if not self.enabled:
            return

        with self._lock:
            if current_state == self._last_state:
                return
            
            # Transition Logic
            prev = self._last_state
            self._last_state = current_state
            
            if current_state == "FAKE" or current_state == "HIGH_RISK":
                # Warning Tone (High Pitch, Rapid)
                self._play_async(1000, 200) # 1kHz for 200ms
                
            elif current_state == "REAL" or current_state == "VERIFIED":
                if prev in ["FAKE", "HIGH_RISK", "UNKNOWN", "NO_FACE"]:
                    # Confirmation Tone (Lower Pitch, Short)
                    self._play_async(600, 100) # 600Hz for 100ms

    def _play_async(self, freq: int, duration: int):
        """Play sound in a separate thread to avoid blocking video pipeline."""
        if self._platform == "windows":
            threading.Thread(target=self._beep, args=(freq, duration), daemon=True).start()
        else:
            # Linux/Mac fallback can be implemented here if needed (e.g. print bell)
            # print('\a') 
            pass

    def disable(self):
        self.enabled = False
