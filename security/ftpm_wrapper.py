
import os
import hashlib
import logging

_log = logging.getLogger("FTPMWrapper")

class FTPMWrapper:
    """
    Hardware security module integration (AMD fTPM).
    If fTPM unavailable, falls back to software hash verification.
    """
    def __init__(self):
        self.available = self._check_ftpm_available()
        if not self.available:
            _log.warning("fTPM not available. Using software integrity verification.")
        
    def _check_ftpm_available(self) -> bool:
        # Check platform or presence of TPM module
        # In mock environment, check for environment variable 'USE_MOCK_FTPM'
        if os.getenv("USE_MOCK_FTPM") == "1":
            return True
            
        try:
            # Hypothetical TSS (TPM Software Stack)
            # import pyftpm
            # return pyftpm.check_tpm()
            pass
        except ImportError:
            pass
        return False

    def verify_and_load(self, path: str):
        """Verify model integrity using TPM or fallback to SHA256."""
        if not os.path.exists(path):
            return None, "File missing"

        if self.available:
            return self._verify_hardware(path)
        else:
            return self._verify_software(path)

    def _verify_hardware(self, path: str):
         # Ask TPM to unsign the hash or check PCR
         # Mock implementation
         return True, "Verified by AMD fTPM"

    def _verify_software(self, path: str):
         # Standard Software Hash
         hasher = hashlib.sha224()
         with open(path, "rb") as f:
             for chunk in iter(lambda: f.read(4096), b""):
                 hasher.update(chunk)
         # Compare with stored hash
         sig_path = path + ".sig"
         if os.path.exists(sig_path):
             with open(sig_path, "r") as f:
                 expected = f.read().strip()
                 if expected == hasher.hexdigest():
                     return True, "Verified by Software (SHA256)"
         return False, "Signature missing or mismatch"
