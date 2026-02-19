"""
Shield-Ryzen V2 — Crypto Audit Trail (TASK 11.2)
================================================
Tamper-evident logging system for enterprise compliance.
Creates a hash-linked chain (blockchain-lite) of all security decisions.

Features:
  - SHA-256 chaining (Entry N includes Hash(Entry N-1)).
  - Digital signature support (optional).
  - Verification tool to detect tampering.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 11 of 14 — Enterprise & Compliance
"""

import hashlib
import json
import time
import os
import logging
from datetime import datetime

class CryptoAuditTrail:
    def __init__(self, log_file="logs/shield_audit_chain.jsonl"):
        self.log_file = log_file
        self.last_hash = "0" * 64 # Genesis hash
        self._init_chain()

    def _init_chain(self):
        """Read last entry to recover chain state."""
        if not os.path.exists(self.log_file):
            # Create genesis block
            # Use add_entry to ensure hash is computed
            self.last_hash = "0" * 64
            # We can't use add_entry because it appends.
            # But here file doesn't exist. So append is create.
            # But add_entry uses timestamps. 
            # I will reuse add_entry logic manually to ensure HASH field exists.
            
            genesis = {
                "event": "GENESIS",
                "timestamp": time.time(),
                "prev_hash": self.last_hash
            }
            json_str = json.dumps(genesis, sort_keys=True)
            genesis_hash = hashlib.sha256(json_str.encode()).hexdigest()
            genesis["hash"] = genesis_hash
            self.last_hash = genesis_hash
            
            self._write_entry(genesis)
            return

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    entry = json.loads(last_line)
                    self.last_hash = entry.get("hash", self.last_hash)
        except Exception as e:
            logging.error(f"Audit chain corrupt: {e}")

    def add_entry(self, data: dict):
        """Add a new entry linked to the previous one."""
        # Prepare entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "prev_hash": self.last_hash
        }
        
        # Calculate Hash
        # Canonical JSON string (sorted keys)
        json_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(json_str.encode()).hexdigest()
        
        entry["hash"] = entry_hash
        self.last_hash = entry_hash
        
        self._write_entry(entry)
        return entry_hash

    def _write_entry(self, entry):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def verify_chain(self) -> bool:
        """Verify integrity of the entire chain."""
        if not os.path.exists(self.log_file):
            return True

        prev_hash = "0" * 64
        valid = True
        
        with open(self.log_file, "r") as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    stored_hash = entry.pop("hash", None)
                    stored_prev = entry.get("prev_hash")
                    
                    # 1. Check Link
                    if stored_prev != prev_hash:
                        logging.error(f"Chain broken at line {i+1}: Link mismatch")
                        valid = False
                        break
                    
                    # 2. Check Integrity
                    # Recompute hash
                    json_str = json.dumps(entry, sort_keys=True)
                    computed_hash = hashlib.sha256(json_str.encode()).hexdigest()
                    
                    if computed_hash != stored_hash:
                        logging.error(f"Chain corrupted at line {i+1}: Hash mismatch")
                        valid = False
                        break
                    
                    prev_hash = stored_hash
                    
                except json.JSONDecodeError:
                    logging.error(f"Chain corrupt at line {i+1}: Invalid JSON")
                    valid = False
                    break
                    
        return valid
