"""
Shield-Ryzen V2 — Structured Audit Logger (TASK 6.5)
====================================================
Logs every security decision, performance metric, and error
in structured JSONL format for post-mortem analysis.

Key Features:
  - JSONL (Newline Delimited JSON) format
  - Thread-safe logging (buffered writes)
  - Levels: AUDIT, WARN, ERROR, DEBUG
  - Memory & FPS tracking per frame

Developer: Inayat Hussain | AMD Slingshot 2026
Part 6 of 14 — Integration & Security
"""

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict
import numpy as np

# Configure standard logger to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ShieldJSONEncoder(json.JSONEncoder):
    """Handles NumPy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

class ShieldLogger:
    """
    Core logging system for Shield-Ryzen V2.
    ...
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.log_dir, "shield_audit.jsonl")
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        
        self.log({
            "event": "system_startup",
            "python_version": sys.version,
            "platform": sys.platform
        }, level="SYSTEM")
        
    def log(self, data: Dict[str, Any], level: str = "AUDIT", event: str = None):
        """Append log entry."""
        timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "level": level,
            "event": event or data.get("event", "unknown"),
            "data": data
        }
        
        # Serialize to JSON line with NumPy support
        line = json.dumps(entry, cls=ShieldJSONEncoder) + "\n"
        
        with self._lock:
            self._file.write(line)
            self._file.flush()
            
    def log_frame(self, frame_data: Dict[str, Any]):
        """Helper for frame processing logs."""
        self.log(frame_data, level="AUDIT", event="frame_processed")
        
    def warn(self, message: str, context: Dict = None):
        """Log structured warning."""
        logging.warning(message)
        self.log({"message": message, "context": context}, level="WARN", event="system_warning")
        
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log structured error with exception details."""
        logging.error(message, **kwargs)
        err_details = str(exception) if exception else None
        self.log({"message": message, "exception": err_details}, level="ERROR", event="system_error")
        
    def close(self):
        """Clean shutdown."""
        with self._lock:
            if not self._file.closed:
                self.log({"message": "Logger shutting down"}, level="SYSTEM", event="system_shutdown")
                self._file.close()

# Use singleton if simple access needed
_logger = None

def get_logger(log_dir="logs"):
    global _logger
    if _logger is None:
        _logger = ShieldLogger(log_dir)
    return _logger
