"""
Shield-Ryzen V2 — Hardware Monitor (TASK 9.2)
=============================================
Tracks system resource usage (CPU, RAM, GPU/NPU proxies).
Ensures low-power, high-efficiency operation on Ryzen mobile platforms.

Features:
  - Process-specific memory/CPU tracking.
  - Thermal monitoring (if supported).
  - FPS vs Wattage estimation (heuristic).

Developer: Inayat Hussain | AMD Slingshot 2026
Part 9 of 14 — AMD Native Optimization
"""

import psutil
import time
import os
import logging
from collections import deque

class HardwareMonitor:
    def __init__(self, history_len=60):
        self.process = psutil.Process()
        self.history_len = history_len
        self.cpu_history = deque(maxlen=history_len)
        self.mem_history = deque(maxlen=history_len)
        self.fps_history = deque(maxlen=history_len)
        self.start_time = time.time()
        
        # Baseline
        self.baseline_power = 0.0 # Setup power model if needed

    def update(self, fps: float):
        """Update metrics for current frame/second."""
        try:
            # CPU usage relative to logical cores
            cpu_percent = self.process.cpu_percent(interval=None) # Non-blocking
            self.cpu_history.append(cpu_percent)
            
            # Memory (RSS) in MB
            mem_mb = self.process.memory_info().rss / 1024 / 1024
            self.mem_history.append(mem_mb)
            
            self.fps_history.append(fps)
            
        except Exception as e:
            logging.warn(f"Monitor update failed: {e}")

    def get_stats(self) -> dict:
        """Return current stats summary."""
        if not self.cpu_history:
            return {"cpu_avg": 0.0, "mem_mb": 0.0, "fps_avg": 0.0}
            
        return {
            "cpu_curr": self.cpu_history[-1],
            "cpu_avg": sum(self.cpu_history) / len(self.cpu_history),
            "mem_mb": self.mem_history[-1],
            "mem_peak": max(self.mem_history),
            "fps_avg": sum(self.fps_history) / max(len(self.fps_history), 1),
            "uptime_sec": time.time() - self.start_time
        }

    def estimate_power(self) -> float:
        """
        Estimate power draw based on heuristic (Ryzen mobile).
        Base: 5W. CPU active: +0.5W per % total CPU.
        Returns: Estimated Watts.
        """
        # Very rough heuristic
        base_w = 5.0
        cpu_load = self.cpu_history[-1] if self.cpu_history else 0
        watts = base_w + (cpu_load * 0.15) # Example
        return watts
