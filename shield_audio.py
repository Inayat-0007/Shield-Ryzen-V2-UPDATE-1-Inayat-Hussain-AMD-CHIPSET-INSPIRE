"""
Shield-Ryzen V2 — Audio Feedback System (TASK 10.3)
===================================================
Provides auditory cues for security state transitions.
Ensures accessibility and immediate user alert.

Features:
  - Background thread for minimal latency.
  - Distinct tones for Verified, High Risk, and Suspicious states.
  - Challenge-Response prompt read-out (Text-To-Speech if available).

Developer: Inayat Hussain | AMD Slingshot 2026
Part 10 of 14 — HUD & Explainability
"""

import threading
import time
import queue
import logging

try:
    import winsound
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

class AudioAlerts:
    def __init__(self, enabled=True):
        self.enabled = enabled and HAS_AUDIO
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.running = True
        if self.enabled:
            self.thread.start()

    def play_alert(self, state: str):
        """Queue an alert for a state transition."""
        if not self.enabled: return
        self.queue.put(state)

    def _worker(self):
        while self.running:
            try:
                state = self.queue.get(timeout=1.0)
                if state == "HIGH_RISK":
                    # Urgent Beep x3
                    for _ in range(3):
                        winsound.Beep(1000, 200)
                        time.sleep(0.1)
                elif state == "VERIFIED":
                    # Ascending Chime
                    winsound.Beep(600, 150)
                    winsound.Beep(800, 150)
                elif state == "SUSPICIOUS":
                    # Low Warning
                    winsound.Beep(400, 300)
                elif state == "CHALLENGE":
                    # Prompt Tone
                    winsound.Beep(1200, 100)
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Audio error: {e}")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
