"""
Shield-Ryzen V2 — Model Verification & Audit (TASK 4.1)
=========================================================
Full cryptographic and architectural audit of the ffpp_c23.pth
checkpoint against the timm XceptionNet (legacy_xception) architecture.

Outputs:
  - model_verification_report.json   (full key mapping + hashes)
  - models/model_signature.sha256    (SHA-256 hash for runtime check)

Architecture Reality:
  The checkpoint has 276 weight keys with "model." prefix.
  The timm legacy_xception (num_classes=2) has 276 keys.
  274 keys match directly after prefix stripping.
  2 keys need renaming: model.last_linear.1.{weight,bias} → fc.{weight,bias}
  This is the classifier head — a known naming difference between the
  FaceForensics++ training code and the timm library.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 4 of 14 — Neural Model Verification & Integrity
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════
# Key equivalence logic
# ═══════════════════════════════════════════════════════════════

# Known FC-head renaming between FF++ training code and timm
_FC_RENAME_MAP: dict[str, str] = {
    "last_linear.1.weight": "fc.weight",
    "last_linear.1.bias": "fc.bias",
}


def _keys_are_equivalent(ckpt_key: str, model_key: str) -> bool:
    """Check if checkpoint key matches model key, accounting for:
    1. Common prefixes: "module.", "model.", "net."
    2. FC head renaming: last_linear.1.* → fc.*
    """
    if ckpt_key == model_key:
        return True

    # Strip known prefixes from checkpoint key
    bare_ck = ckpt_key
    for prefix in ("module.", "model.", "net."):
        if bare_ck.startswith(prefix):
            bare_ck = bare_ck[len(prefix):]
            break

    if bare_ck == model_key:
        return True

    # Check FC head rename
    if bare_ck in _FC_RENAME_MAP and _FC_RENAME_MAP[bare_ck] == model_key:
        return True

    return False


# ═══════════════════════════════════════════════════════════════
# Architecture introspection
# ═══════════════════════════════════════════════════════════════

def _count_depthwise_separable_convs(model: torch.nn.Module) -> dict:
    """Count and describe the depthwise separable convolution layers.

    In Xception, a SeparableConv2d consists of:
      - conv1: depthwise convolution (groups == in_channels)
      - pointwise: 1x1 convolution
    """
    depthwise = []
    pointwise = []
    standard = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.groups == module.in_channels and module.groups > 1:
                depthwise.append({
                    "name": name,
                    "in_ch": module.in_channels,
                    "out_ch": module.out_channels,
                    "kernel": list(module.kernel_size),
                    "groups": module.groups,
                })
            elif module.kernel_size == (1, 1):
                pointwise.append({
                    "name": name,
                    "in_ch": module.in_channels,
                    "out_ch": module.out_channels,
                })
            else:
                standard.append({
                    "name": name,
                    "in_ch": module.in_channels,
                    "out_ch": module.out_channels,
                    "kernel": list(module.kernel_size),
                })

    return {
        "depthwise_conv_count": len(depthwise),
        "pointwise_conv_count": len(pointwise),
        "standard_conv_count": len(standard),
        "separable_pair_count": min(len(depthwise), len(pointwise)),
        "total_conv_count": len(depthwise) + len(pointwise) + len(standard),
        "depthwise_layers": depthwise,
        "pointwise_layers": pointwise,
        "standard_layers": standard,
    }


def _count_parameters(model: torch.nn.Module) -> dict:
    """Count total and per-block parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    block_counts = {}
    for name, p in model.named_parameters():
        block = name.split(".")[0]
        block_counts[block] = block_counts.get(block, 0) + p.numel()

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "per_block": block_counts,
    }


# ═══════════════════════════════════════════════════════════════
# Reference output generation
# ═══════════════════════════════════════════════════════════════

def _generate_reference_output(
    model: torch.nn.Module,
    state_dict: dict,
    device: torch.device,
) -> dict:
    """Run a deterministic reference input through the loaded model
    and record the output for future regression testing.

    Reference input: all-zeros tensor (1, 3, 299, 299) — deterministic.
    """
    model.eval()
    ref_input = torch.zeros(1, 3, 299, 299, device=device)

    with torch.no_grad():
        logits = model(ref_input)
        probs = torch.softmax(logits, dim=1) if logits.shape[-1] == 2 else logits

    return {
        "reference_input": "zeros(1, 3, 299, 299)",
        "output_raw": logits.cpu().numpy().tolist(),
        "output_shape": list(logits.shape),
        "fake_prob": float(probs[0, 0]),
        "real_prob": float(probs[0, 1]),
    }


# ═══════════════════════════════════════════════════════════════
# Main verification function
# ═══════════════════════════════════════════════════════════════

def verify_checkpoint(
    checkpoint_path: str,
    output_dir: Optional[str] = None,
) -> dict:
    """Full audit of the ffpp_c23 checkpoint.

    Steps:
      1. SHA-256 hash computation
      2. Load checkpoint safely
      3. Create reference timm model (legacy_xception, num_classes=2)
      4. Map every key with equivalence checking
      5. Introspect architecture (count depthwise separable convs)
      6. Generate reference output on deterministic input
      7. Save full report + signature file

    Returns:
        Full verification report dict.
    """
    if output_dir is None:
        output_dir = str(_PROJECT_ROOT)

    print(f"{'=' * 60}")
    print(f"  PART 4 — MODEL VERIFICATION AUDIT")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Step 1: Hash ──────────────────────────────────────────
    print("[1/7] Computing SHA-256 hash...")
    sha256 = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    file_hash = sha256.hexdigest()
    file_size = os.path.getsize(checkpoint_path)
    print(f"      Hash: {file_hash}")
    print(f"      Size: {file_size:,} bytes")

    # ── Step 2: Load checkpoint ───────────────────────────────
    print("[2/7] Loading checkpoint...")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception:
        print("      weights_only=True failed, using fallback...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle nested state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    ckpt_keys = list(state_dict.keys())
    print(f"      Loaded {len(ckpt_keys)} keys")

    # ── Step 3: Reference model ───────────────────────────────
    print("[3/7] Creating reference model (timm.legacy_xception, num_classes=2)...")
    ref_model = timm.create_model("xception", pretrained=False, num_classes=2)
    model_keys = list(ref_model.state_dict().keys())
    print(f"      Reference model has {len(model_keys)} keys")

    # ── Step 4: Key mapping ───────────────────────────────────
    print("[4/7] Mapping checkpoint keys to model keys...")
    matched_keys = []
    unmatched_ckpt_keys = []
    missing_model_keys = set(model_keys)

    for ck in ckpt_keys:
        found = False
        for mk in model_keys:
            if _keys_are_equivalent(ck, mk):
                matched_keys.append({"checkpoint": ck, "model": mk})
                missing_model_keys.discard(mk)
                found = True
                break
        if not found:
            unmatched_ckpt_keys.append(ck)

    missing_model_keys_list = sorted(missing_model_keys)
    success_rate = (len(matched_keys) / len(model_keys) * 100) if model_keys else 0.0

    print(f"      Matched:     {len(matched_keys)}/{len(model_keys)} ({success_rate:.1f}%)")
    print(f"      Unmatched:   {len(unmatched_ckpt_keys)}")
    print(f"      Missing:     {len(missing_model_keys_list)}")

    # ── Step 5: Architecture introspection ────────────────────
    print("[5/7] Introspecting architecture (depthwise separable convolutions)...")
    conv_info = _count_depthwise_separable_convs(ref_model)
    param_info = _count_parameters(ref_model)
    print(f"      Total parameters:  {param_info['total_parameters']:,}")
    print(f"      Depthwise convs:   {conv_info['depthwise_conv_count']}")
    print(f"      Pointwise convs:   {conv_info['pointwise_conv_count']}")
    print(f"      Standard convs:    {conv_info['standard_conv_count']}")
    print(f"      Separable pairs:   {conv_info['separable_pair_count']}")

    # ── Step 6: Reference output ──────────────────────────────
    print("[6/7] Generating reference output on deterministic input...")
    # Load weights into the reference model for reference output
    clean_sd = {}
    for ck, v in state_dict.items():
        bare = ck
        for prefix in ("module.", "model.", "net."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break
        if bare in _FC_RENAME_MAP:
            bare = _FC_RENAME_MAP[bare]
        clean_sd[bare] = v

    ref_model.load_state_dict(clean_sd, strict=False)
    ref_model.eval()
    ref_output = _generate_reference_output(ref_model, clean_sd, torch.device("cpu"))
    print(f"      Output shape: {ref_output['output_shape']}")
    print(f"      Fake prob:    {ref_output['fake_prob']:.6f}")
    print(f"      Real prob:    {ref_output['real_prob']:.6f}")

    # ── Step 7: Build report ──────────────────────────────────
    print("[7/7] Saving report...")

    report = {
        "verification_version": "4.0",
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": checkpoint_path,
        "file_size_bytes": file_size,
        "checkpoint_hash_sha256": file_hash,
        "total_checkpoint_keys": len(ckpt_keys),
        "total_model_keys": len(model_keys),
        "matched_key_count": len(matched_keys),
        "unmatched_checkpoint_key_count": len(unmatched_ckpt_keys),
        "missing_model_key_count": len(missing_model_keys_list),
        "mapping_success_rate": round(success_rate, 2),
        "fc_head_rename": {
            "description": "FC head uses different naming between FF++ training code and timm",
            "checkpoint_names": ["model.last_linear.1.weight", "model.last_linear.1.bias"],
            "model_names": ["fc.weight", "fc.bias"],
            "handled": True,
        },
        "architecture": {
            "family": "XceptionNet",
            "timm_variant": "legacy_xception",
            "num_classes": 2,
            "input_shape": [1, 3, 299, 299],
            "output_shape": [1, 2],
            "mechanism": (
                "Depthwise separable convolutions decompose standard convolutions "
                "into a depthwise convolution (per-channel spatial filtering) followed "
                "by a pointwise 1x1 convolution (cross-channel mixing). The LEARNED "
                "FILTERS within these convolutions detect microscopic digital artifacts "
                "and texture inconsistencies specific to GAN/Diffusion-generated faces."
            ),
            "conv_counts": {
                "depthwise": conv_info["depthwise_conv_count"],
                "pointwise": conv_info["pointwise_conv_count"],
                "standard": conv_info["standard_conv_count"],
                "separable_pairs": conv_info["separable_pair_count"],
            },
            "parameters": param_info,
        },
        "normalization": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "range": [-1.0, 1.0],
            "formula": "(pixel / 255.0 - 0.5) / 0.5",
        },
        "output_mapping": {
            "index_0": "FAKE probability",
            "index_1": "REAL probability",
            "activation": "softmax (in ShieldXception.forward)",
        },
        "reference_output": ref_output,
        "matched_keys": matched_keys,
        "unmatched_checkpoint_keys": unmatched_ckpt_keys,
        "missing_model_keys": missing_model_keys_list,
    }

    # Save report
    report_path = os.path.join(output_dir, "model_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"      Report: {report_path}")

    # Save reference output separately for Part 14 regression
    ref_path = os.path.join(output_dir, "models", "reference_output.json")
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref_output, f, indent=2)
    print(f"      Reference output: {ref_path}")

    # Save SHA-256 signature
    sig_path = os.path.join(output_dir, "models", "model_signature.sha256")
    with open(sig_path, "w", encoding="utf-8") as f:
        f.write(file_hash)
    print(f"      Signature: {sig_path}")

    # Summary
    all_ok = (
        success_rate >= 99.0
        and len(missing_model_keys_list) == 0
        and ref_output["output_shape"] == [1, 2]
    )

    print(f"\n{'=' * 60}")
    if all_ok:
        print(f"  ✅ VERIFICATION: PASS")
    else:
        print(f"  ⚠️  VERIFICATION: ISSUES FOUND")
        if success_rate < 99.0:
            print(f"     - Key mapping below 99%: {success_rate:.1f}%")
        if missing_model_keys_list:
            print(f"     - {len(missing_model_keys_list)} model keys not matched")
    print(f"  Hash:       {file_hash[:16]}...")
    print(f"  Keys:       {len(matched_keys)}/{len(model_keys)} matched")
    print(f"  Parameters: {param_info['total_parameters']:,}")
    print(f"  Separable:  {conv_info['separable_pair_count']} depthwise separable conv pairs")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    try:
        verify_checkpoint("ffpp_c23.pth")
    except Exception as e:
        print(f"ERROR: Verification failed — {e}")
        sys.exit(1)
