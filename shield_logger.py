
import os
import json
import time
import logging

_log = logging.getLogger("ShieldLogger")

class ShieldLogger:
    """Structured JSONL logger for Shield-Ryzen audit trail.
    
    Logs every processed frame with timing, memory, and detection details.
    Thread-safe enough for the main engine loop (single writer).
    """

    def __init__(self, log_path: str = "logs/shield_audit.jsonl"):
        """Initialize logger, creating directory if needed.
        
        Args:
            log_path: Path to the JSONL log file.
        """
        self.log_path = log_path
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, "a", encoding="utf-8")
            _log.info(f"Audit log opened at {log_path}")
        except Exception as e:
            _log.error(f"Failed to open audit log {log_path}: {e}")
            self.log_file = None

    def log_frame(self, data: dict) -> None:
        """Write a dictionary as a JSON line to the log file.
        
        Args:
            data: Dictionary of data to log. Timestamp added if missing.
        """
        if self.log_file is None:
            return

        if "timestamp" not in data:
            data["timestamp"] = time.time()

        try:
            entry = json.dumps(data)
            self.log_file.write(entry + "\n")
            self.log_file.flush()
        except Exception as e:
            _log.error(f"Failed to write log entry: {e}")

    def warn(self, message: str) -> None:
        """Log a warning message to the audit trail.
        
        Args:
            message: Warning content.
        """
        self.log_frame({
            "level": "WARN",
            "message": message,
            "timestamp": time.time()
        })
        _log.warning(message)

    def close(self) -> None:
        """Close the log file handler."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
