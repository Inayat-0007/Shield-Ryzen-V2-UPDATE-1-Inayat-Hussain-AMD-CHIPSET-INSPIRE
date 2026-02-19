"""
Shield-Ryzen V2 — ArcFace Re-ID Plugin (TASK 11.1)
==================================================
Enterprise Identity Verification using ArcFace embeddings.
Matches verified real faces against an encrypted employee database.

Features:
  - Generates 512-d (or 128-d) embeddings from face crops.
  - Cosine similarity matching.
  - Encrypted local storage (AES-256 via shield_crypto).
  - Strict privacy: No cloud transmission.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 11 of 14 — Enterprise & Compliance
"""

import numpy as np
import cv2
import os
import json
import logging
import pickle

# Add project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_plugin import ShieldPlugin
from shield_crypto import encrypt, decrypt

class ArcFacePlugin(ShieldPlugin):
    name = "face_reid"
    tier = "identity"

    def __init__(self, model_path="models/arcface_w600k_r50.onnx", db_path="secure_data/employee_db.enc"):
        self.model_path = model_path
        self.db_path = db_path
        self.users = {}
        self.threshold = 0.4 # Cosine distance threshold
        
        # Load DB
        self._load_db()
        
        # Load Model
        self.session = None
        try:
            import onnxruntime as ort
            if os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            else:
                logging.warn(f"ArcFace model not found at {model_path}. Running in MOCK mode.")
        except Exception as e:
            logging.error(f"Failed to init ArcFace: {e}")

    def _load_db(self):
        """Load encrypted user database."""
        if not os.path.exists(self.db_path):
            return

        try:
            with open(self.db_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt
            json_str = decrypt(encrypted_data)
            if json_str:
                data = json.loads(json_str)
                # Convert lists back to arrays
                self.users = {uid: np.array(emb, dtype=np.float32) for uid, emb in data.items()}
        except Exception as e:
            logging.error(f"Failed to load ID DB: {e}")

    def _save_db(self):
        """Encrypt and save user database."""
        try:
            # Convert arrays to lists
            data = {uid: emb.tolist() for uid, emb in self.users.items()}
            json_str = json.dumps(data)
            
            encrypted_data = encrypt(json_str)
            
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, "wb") as f:
                f.write(encrypted_data)
        except Exception as e:
            logging.error(f"Failed to save ID DB: {e}")

    def enroll_user(self, face_crop, user_id):
        """Generate embedding and save to DB."""
        embedding = self._get_embedding(face_crop)
        if embedding is not None:
            self.users[user_id] = embedding
            self._save_db()
            return True
        return False

    def analyze(self, face, frame: np.ndarray) -> dict:
        """
        Verify identity.
        Returns 'MATCH' if verified user, 'UNKNOWN' otherwise.
        """
        if face.state == "FAKE" or face.confidence < 0.8:
            return {"verdict": "SKIP", "confidence": 0.0, "name": self.name, "explanation": "Face not trusted"}
            
        emb = self._get_embedding(face.face_crop_raw) # Use raw crop? Or aligned?
        # Model usually expects 112x112 aligned. 
        # ShieldFacePipeline aligns to 299x299.
        # We resize to 112x112 here.
        
        if emb is None:
             return {"verdict": "ERROR", "confidence": 0.0, "name": self.name, "explanation": "Embedding failed"}

        best_score = -1.0
        best_uid = "Unknown"
        
        for uid, ref_emb in self.users.items():
            # Cosine Similarity
            score = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb) + 1e-6)
            if score > best_score:
                best_score = score
                best_uid = uid
        
        matched = best_score > self.threshold
        
        return {
            "verdict": "MATCH" if matched else "UNKNOWN_USER",
            "confidence": float(best_score),
            "name": self.name,
            "explanation": f"ID: {best_uid} ({best_score:.2f})"
        }

    def _get_embedding(self, face_crop):
        if self.session is None:
            # Mock mode: Return random vector
            return np.random.rand(512).astype(np.float32)
        
        try:
            # Resize to 112x112 (ArcFace standard)
            img = cv2.resize(face_crop, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = (img - 127.5) / 128.0
            img = np.expand_dims(img, axis=0)
            
            return self.session.run(None, {self.input_name: img})[0][0]
        except Exception:
            return None
