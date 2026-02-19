"""
Shield-Ryzen V2 — Enterprise Integration Tests (TASK 11.6)
==========================================================
Verifies Identity Management, Audit Trail Integrity, and Compliance Data Handling.
Ensures critical enterprise features (re-ID, chain-of-custody) function correctly.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 11 of 14 — Enterprise & Compliance
"""

import unittest
import shutil
import json
import os
import sys
import numpy as np
from unittest.mock import MagicMock

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.arcface_reid import ArcFacePlugin
from security.audit_trail import CryptoAuditTrail
from shield_engine import FaceResult

class TestEnterpriseFeatures(unittest.TestCase):
    
    def setUp(self):
        # Temp dir for test
        self.test_dir = "tests/temp_enterprise"
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.audit_file = os.path.join(self.test_dir, "test_audit.jsonl")
        self.db_path = os.path.join(self.test_dir, "test_db.enc")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # 1. ArcFace Re-ID
    def test_arcface_enrollment_and_recognition(self):
        # Using mock model (since ONNX likely missing in test env)
        plugin = ArcFacePlugin(model_path="invalid/path.onnx", db_path=self.db_path)
        
        # Determine if running in mock mode
        # Implementation logs warning but session is None
        self.assertIsNone(plugin.session)
        
        # Mock embedding Generation
        # Assume _get_embedding returns random vector
        # Enrollment
        fake_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        success = plugin.enroll_user(fake_crop, "user_123")
        self.assertTrue(success)
        self.assertIn("user_123", plugin.users)
        
        # Verify persistence
        plugin._save_db()
        self.assertTrue(os.path.exists(self.db_path))
        
        # Re-load
        plugin2 = ArcFacePlugin(model_path="invalid/path.onnx", db_path=self.db_path)
        self.assertIn("user_123", plugin2.users)
        
        # Recognition
        # Mock face with state=REAL
        mock_face = MagicMock()
        mock_face.face_crop_raw = fake_crop
        mock_face.state = "REAL"
        mock_face.confidence = 0.95
        
        # The embedding generated is random, so similarity will be random.
        # But we can mock _get_embedding to return same vector
        embedding = plugin.users["user_123"]
        plugin._get_embedding = MagicMock(return_value=embedding)
        
        res = plugin.analyze(mock_face, None)
        self.assertEqual(res["verdict"], "MATCH")
        self.assertIn("user_123", res["explanation"])

    # 2. Audit Trail
    def test_audit_trail_chain_integrity(self):
        audit = CryptoAuditTrail(log_file=self.audit_file)
        
        # Add entries
        h1 = audit.add_entry({"event": "login", "user": "admin"})
        h2 = audit.add_entry({"event": "scan", "result": "FAKE"})
        
        self.assertTrue(os.path.exists(self.audit_file))
        
        # Verify
        self.assertTrue(audit.verify_chain())
        
        # Tamper
        with open(self.audit_file, "r") as f:
            lines = f.readlines()
        
        # Modify line 1 (Genesis is 0, login is 1)
        # Actually line 0 is GENESIS if file didn't exist?
        # Check _init_chain: writes GENESIS if not exists.
        # So file has: GENESIS, login, scan.
        line1 = json.loads(lines[1])
        line1["data"]["user"] = "hacker"
        # Since hash is part of line, modify content changes hash.
        # But we keep old hash?
        # If we change content, recomputed hash != stored hash.
        lines[1] = json.dumps(line1) + "\n"
        
        with open(self.audit_file, "w") as f:
            f.writelines(lines)
            
        # Verify fail
        self.assertFalse(audit.verify_chain())

    def test_compliance_consent_mechanism(self):
        # We check Default Config in ShieldEngine has explicit Opt-In default
        from shield_engine import DEFAULT_CONFIG
        self.assertFalse(DEFAULT_CONFIG["enable_challenge_response"], 
                         "Challenge Response (Biometric Collection) must be OPT-IN by default for GDPR/BIPA compliance")
        self.assertFalse(DEFAULT_CONFIG["enable_lip_sync"],
                         "Lip Sync (Audio Collection) must be OPT-IN by default")

if __name__ == '__main__':
    unittest.main()
