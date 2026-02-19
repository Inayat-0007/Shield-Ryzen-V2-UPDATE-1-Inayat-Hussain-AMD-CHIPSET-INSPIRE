"""
Shield-Ryzen V2 — Encrypted Biometric Pipeline (TASK 6.3)
=========================================================
In-memory AES-256 encryption for transient biometric data.
Ensures that RAM dumps yield no usable face crops or landmarks.

Privacy Promise:
  - Ephemeral keys (generated per session, never stored)
  - AES-256-GCM (Authenticated Encryption)
  - Immediate encryption of sensitive numpy arrays
  - Decryption only during active processing
  - Secure wiping on exit

Developer: Inayat Hussain | AMD Slingshot 2026
Part 6 of 14 — Integration & Security
"""

import os
import sys
import secrets
import logging
import pickle
from typing import Optional, Union, Tuple

# Try importing cryptography, fallback to simple XOR obfuscation if missing
# (For a competition prototype without heavy dependencies, but we aim for AES)
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_AES = True
except ImportError:
    HAS_AES = False

class BiometricEncryptor:
    """
    Manages ephemeral encryption keys and secures biometric payloads.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("BiometricEncryptor")
        self._key = None
        self._aes = None
        
        if HAS_AES:
            # Generate 256-bit ephemeral key
            self._key = AESGCM.generate_key(bit_length=256)
            self._aes = AESGCM(self._key)
            self.logger.info("Initialized secure AES-256-GCM pipeline.")
        else:
            self.logger.warning("cryptography module missing! Using fallback obfuscation.")
            # Fallback key for XOR
            self._key = secrets.token_bytes(32)

    def encrypt_data(self, data: object) -> bytes:
        """
        Serialize and encrypt any Python object (numpy array, dict, etc).
        Returns: iv + ciphertext + tag
        """
        payload = pickle.dumps(data)
        
        if HAS_AES:
            iv = os.urandom(12) # 96-bit nonce for GCM
            ciphertext = self._aes.encrypt(iv, payload, None) # No AAD
            return iv + ciphertext
        else:
            # Fallback: Simple XOR (Not secure against determined attacker, but obfuscates RAM)
            key_len = len(self._key)
            encrypted = bytearray(len(payload))
            for i in range(len(payload)):
                encrypted[i] = payload[i] ^ self._key[i % key_len]
            return bytes(encrypted)

    def decrypt_data(self, blob: bytes) -> object:
        """
        Decrypt and deserialize data.
        """
        if HAS_AES:
            iv = blob[:12]
            ciphertext = blob[12:]
            try:
                plaintext = self._aes.decrypt(iv, ciphertext, None)
                return pickle.loads(plaintext)
            except Exception as e:
                self.logger.error(f"Decryption failed: {e}")
                return None
        else:
            # Fallback XOR
            key_len = len(self._key)
            decrypted = bytearray(len(blob))
            for i in range(len(blob)):
                decrypted[i] = blob[i] ^ self._key[i % key_len]
            return pickle.loads(bytes(decrypted))

    def secure_wipe(self):
        """
        Best-effort clearing of key from memory.
        (Python doesn't guarantee memory clearing due to immutability/GC, 
         but we dereference immediately).
        """
        self._key = None
        self._aes = None
        self.logger.info("Biometric keys purged.")

# Singleton instance
_encryptor = BiometricEncryptor()

def encrypt(data):
    return _encryptor.encrypt_data(data)

def decrypt(blob):
    return _encryptor.decrypt_data(blob)

def secure_wipe():
    _encryptor.secure_wipe()
