"""
Shield-Ryzen V2 -- Model Verification & Audit
==============================================
Verifies the integrity and architecture of the ffpp_c23.pth checkpoint.
Checks if the checkpoint keys match a standard XceptionNet model from timm.
Generates a full verification report.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 4 of 12 -- Neural Model Verification
"""

import hashlib
import json
import os
import sys

import timm
import torch

def _keys_are_equivalent(ckpt_key: str, model_key: str) -> bool:
    """Check if checkpoint key matches model key, handling potential prefix differences."""
    # Common prefixes in checkpoints: "module.", "model.", "net."
    if ckpt_key == model_key:
        return True
    
    prefixes = ["module.", "model.", "net."]
    for p in prefixes:
        if ckpt_key.startswith(p) and ckpt_key[len(p):] == model_key:
            return True
            
    return False

def verify_checkpoint(checkpoint_path: str) -> dict:
    """Loads the checkpoint and performs full audit against Xception architecture."""
    print(f"Verifying checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Calculate SHA256 hash
    sha256_hash = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()
    file_size = os.path.getsize(checkpoint_path)

    # Load Checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        # Fallback for older PyTorch versions or complex pickles if weights_only fails
        print(f"Warning: weights_only load failed ({e}), trying unsafe load...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats (state_dict inside 'state_dict' key or direct)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    ckpt_keys = list(state_dict.keys())

    # Create Reference Model (Xception)
    # num_classes=2 for Real vs Fake
    model = timm.create_model("xception", pretrained=False, num_classes=2)
    model_keys = list(model.state_dict().keys())

    report = {
        "checkpoint_path": checkpoint_path,
        "file_size_bytes": file_size,
        "checkpoint_hash_sha256": file_hash,
        "total_checkpoint_keys": len(ckpt_keys),
        "total_model_keys": len(model_keys),
        "matched_keys": [],
        "unmatched_checkpoint_keys": [],
        "missing_model_keys": [],
        "verification_timestamp": "",
    }

    # Match Keys
    # We remove keys from the 'missing' list as we find matches
    missing_keys = set(model_keys)
    
    for ck in ckpt_keys:
        matched = False
        for mk in model_keys:
            if _keys_are_equivalent(ck, mk):
                report["matched_keys"].append({"checkpoint": ck, "model": mk})
                if mk in missing_keys:
                    missing_keys.remove(mk)
                matched = True
                break
        if not matched:
            report["unmatched_checkpoint_keys"].append(ck)

    report["missing_model_keys"] = list(missing_keys)
    
    match_count = len(report["matched_keys"])
    success_rate = (match_count / len(model_keys) * 100) if model_keys else 0.0
    
    report["mapping_success_rate"] = round(success_rate, 2)
    
    print(f"  Hash: {file_hash}")
    print(f"  Keys: {len(ckpt_keys)} checkpoint vs {len(model_keys)} model")
    print(f"  Matches: {match_count}")
    print(f"  Success Rate: {success_rate:.1f}%")

    if success_rate < 90.0:
        print("WARNING: Model architecture mismatch likely!")
    else:
        print("SUCCESS: Checkpoint matches Xception architecture.")

    # Save Report
    output_file = "model_verification_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Audit report saved to {output_file}")
    
    # Save the hash to a signature file for runtime verification
    sig_dir = "models"
    os.makedirs(sig_dir, exist_ok=True)
    sig_path = os.path.join(sig_dir, "model_signature.sha256")
    with open(sig_path, "w") as f:
        f.write(file_hash)
    print(f"Model signature saved to {sig_path}")

    return report

if __name__ == "__main__":
    try:
        verify_checkpoint("ffpp_c23.pth")
    except Exception as e:
        print(f"ERROR: Verification failed -- {e}")
        sys.exit(1)
