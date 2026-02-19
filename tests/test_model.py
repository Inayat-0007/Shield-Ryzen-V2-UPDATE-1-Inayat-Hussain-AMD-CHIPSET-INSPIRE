"""
Shield-Ryzen V2 — Model Verification Tests (TASK 4.6)
======================================================
8 tests covering checkpoint integrity, key mapping,
model shape verification, reference output regression,
and tamper detection.

Tests:
  1. test_checkpoint_key_count_matches_actual
  2. test_all_keys_mapped_successfully
  3. test_no_missing_model_keys
  4. test_model_hash_matches_expected
  5. test_model_input_shape_correct
  6. test_model_output_shape_correct
  7. test_model_output_on_reference_image
  8. test_tampered_model_raises_security_error

Developer: Inayat Hussain | AMD Slingshot 2026
Part 4 of 14 — Neural Model Verification & Integrity
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import timm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shield_xception import (
    ShieldXception,
    SecurityError,
    ModelTamperingError,
    ModelShapeError,
    MODEL_EXPECTED_HASH,
    MODEL_EXPECTED_KEY_COUNT,
    MODEL_INPUT_SHAPE,
    MODEL_OUTPUT_SHAPE,
    load_model_with_verification,
    _compute_file_hash,
)
from verify_model import verify_checkpoint, _keys_are_equivalent


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════

CHECKPOINT_PATH = str(Path(__file__).resolve().parent.parent / "ffpp_c23.pth")
SIGNATURE_PATH = str(Path(__file__).resolve().parent.parent / "models" / "model_signature.sha256")
REPORT_PATH = str(Path(__file__).resolve().parent.parent / "model_verification_report.json")
REF_OUTPUT_PATH = str(Path(__file__).resolve().parent.parent / "models" / "reference_output.json")

# Whether the actual checkpoint file is available (skip heavy tests if not)
HAS_CHECKPOINT = os.path.exists(CHECKPOINT_PATH)
HAS_SIGNATURE = os.path.exists(SIGNATURE_PATH)


# ═══════════════════════════════════════════════════════════════
#  Test 1: Key count matches actual checkpoint
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not available")
def test_checkpoint_key_count_matches_actual():
    """Verify the actual checkpoint has exactly MODEL_EXPECTED_KEY_COUNT keys."""
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    actual_count = len(state_dict)
    assert actual_count == MODEL_EXPECTED_KEY_COUNT, (
        f"Checkpoint has {actual_count} keys, expected {MODEL_EXPECTED_KEY_COUNT}"
    )


# ═══════════════════════════════════════════════════════════════
#  Test 2: All checkpoint keys can be mapped to model keys
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not available")
def test_all_keys_mapped_successfully():
    """Every checkpoint key must map to a model key via _keys_are_equivalent.

    The FF++ checkpoint has 276 keys with 'model.' prefix.
    274 map directly, 2 need FC head renaming (handled by _keys_are_equivalent).
    After FC rename handling, ALL 276 should be matched.
    """
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Create reference model
    ref_model = timm.create_model("xception", pretrained=False, num_classes=2)
    model_keys = list(ref_model.state_dict().keys())

    unmatched = []
    for ck in state_dict.keys():
        found = any(_keys_are_equivalent(ck, mk) for mk in model_keys)
        if not found:
            unmatched.append(ck)

    assert len(unmatched) == 0, (
        f"{len(unmatched)} checkpoint keys could not be mapped: {unmatched}"
    )


# ═══════════════════════════════════════════════════════════════
#  Test 3: No missing model keys (all timm keys have a checkpoint source)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not available")
def test_no_missing_model_keys():
    """Every key in the reference timm model must be provided by the checkpoint."""
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    ckpt_keys = list(state_dict.keys())

    ref_model = timm.create_model("xception", pretrained=False, num_classes=2)
    model_keys = list(ref_model.state_dict().keys())

    missing = []
    for mk in model_keys:
        found = any(_keys_are_equivalent(ck, mk) for ck in ckpt_keys)
        if not found:
            missing.append(mk)

    assert len(missing) == 0, (
        f"{len(missing)} model keys have no checkpoint source: {missing}"
    )


# ═══════════════════════════════════════════════════════════════
#  Test 4: SHA-256 hash matches expected
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not available")
def test_model_hash_matches_expected():
    """SHA-256 hash of ffpp_c23.pth must match the compiled constant."""
    actual_hash = _compute_file_hash(CHECKPOINT_PATH)

    assert actual_hash == MODEL_EXPECTED_HASH, (
        f"Hash mismatch!\n"
        f"Expected: {MODEL_EXPECTED_HASH}\n"
        f"Actual:   {actual_hash}"
    )

    # Also verify signature file matches
    if HAS_SIGNATURE:
        with open(SIGNATURE_PATH, "r") as f:
            sig_hash = f.read().strip()
        assert sig_hash == MODEL_EXPECTED_HASH, (
            f"Signature file hash does not match compiled constant"
        )


# ═══════════════════════════════════════════════════════════════
#  Test 5: Model input shape is correct
# ═══════════════════════════════════════════════════════════════

def test_model_input_shape_correct():
    """ShieldXception must accept (1, 3, 299, 299) input without error."""
    model = ShieldXception()
    model.eval()

    test_input = torch.randn(*MODEL_INPUT_SHAPE)

    with torch.no_grad():
        output = model(test_input)

    # Model accepted the input — now check output exists and has correct dims
    assert output is not None, "Model returned None"
    assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"
    assert output.shape[0] == 1, f"Expected batch=1, got {output.shape[0]}"

    # Reject wrong channel count — 1 channel should fail
    wrong_input = torch.randn(1, 1, 299, 299)
    with pytest.raises(Exception):
        model(wrong_input)


# ═══════════════════════════════════════════════════════════════
#  Test 6: Model output shape is correct
# ═══════════════════════════════════════════════════════════════

def test_model_output_shape_correct():
    """Output must be (1, 2) softmax probabilities summing to ~1.0."""
    model = ShieldXception()
    model.eval()

    test_input = torch.randn(*MODEL_INPUT_SHAPE)

    with torch.no_grad():
        output = model(test_input)

    # Shape check
    assert output.shape == torch.Size(list(MODEL_OUTPUT_SHAPE)), (
        f"Expected output shape {MODEL_OUTPUT_SHAPE}, got {tuple(output.shape)}"
    )

    # Softmax property: sums to 1.0
    sum_probs = output.sum().item()
    assert abs(sum_probs - 1.0) < 1e-5, (
        f"Output should sum to 1.0 (softmax), got {sum_probs:.6f}"
    )

    # All probabilities in [0, 1]
    assert (output >= 0).all() and (output <= 1).all(), (
        "Output probabilities must be in [0, 1]"
    )

    # Two outputs: [fake_prob, real_prob]
    assert output.shape[1] == 2, (
        f"Expected 2 output classes, got {output.shape[1]}"
    )


# ═══════════════════════════════════════════════════════════════
#  Test 7: Model output on reference (deterministic) input
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not available")
def test_model_output_on_reference_image():
    """Model output on zeros(1,3,299,299) must match recorded reference output.

    This is a regression test — if the model weights change,
    the reference output will no longer match.
    """
    # Load model with verified weights
    model = ShieldXception()
    model.eval()

    state_dict = load_model_with_verification(CHECKPOINT_PATH, torch.device("cpu"))
    model.model.load_state_dict(state_dict, strict=False)

    # Deterministic reference input: all zeros
    ref_input = torch.zeros(*MODEL_INPUT_SHAPE)

    with torch.no_grad():
        output = model(ref_input)

    fake_prob = float(output[0, 0])
    real_prob = float(output[0, 1])

    # The output must be a valid softmax distribution
    assert abs(fake_prob + real_prob - 1.0) < 1e-5, "Output must sum to 1.0"

    # Check against saved reference if available
    if os.path.exists(REF_OUTPUT_PATH):
        with open(REF_OUTPUT_PATH, "r") as f:
            ref = json.load(f)

        ref_fake = ref["fake_prob"]
        ref_real = ref["real_prob"]

        # Allow small floating point tolerance (1e-4)
        assert abs(fake_prob - ref_fake) < 1e-4, (
            f"Fake prob regression: expected {ref_fake:.6f}, got {fake_prob:.6f}"
        )
        assert abs(real_prob - ref_real) < 1e-4, (
            f"Real prob regression: expected {ref_real:.6f}, got {real_prob:.6f}"
        )
    else:
        # No reference file yet — just verify the output is sane
        assert 0.0 <= fake_prob <= 1.0, f"Fake prob out of range: {fake_prob}"
        assert 0.0 <= real_prob <= 1.0, f"Real prob out of range: {real_prob}"


# ═══════════════════════════════════════════════════════════════
#  Test 8: Tampered model raises ModelTamperingError
# ═══════════════════════════════════════════════════════════════

def test_tampered_model_raises_security_error():
    """A model file with a different hash must raise ModelTamperingError."""
    # Create a tampered model file (random bytes)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp.write(b"TAMPERED MODEL DATA " * 100)
        tampered_path = tmp.name

    try:
        # The signature file stores the real hash; loading the tampered file
        # should detect the mismatch
        with pytest.raises(ModelTamperingError) as exc_info:
            load_model_with_verification(tampered_path, skip_hash=False)

        # Verify error message contains useful info
        error_msg = str(exc_info.value)
        assert "INTEGRITY VIOLATION" in error_msg
        assert "Expected:" in error_msg
        assert "Got:" in error_msg
    finally:
        os.unlink(tampered_path)


# ═══════════════════════════════════════════════════════════════
#  BONUS: Test _keys_are_equivalent covers all cases
# ═══════════════════════════════════════════════════════════════

def test_key_equivalence_logic():
    """Verify the key mapping handles all prefix patterns and FC rename."""
    # Direct match
    assert _keys_are_equivalent("conv1.weight", "conv1.weight")

    # Prefix stripping
    assert _keys_are_equivalent("model.conv1.weight", "conv1.weight")
    assert _keys_are_equivalent("module.conv1.weight", "conv1.weight")
    assert _keys_are_equivalent("net.conv1.weight", "conv1.weight")

    # FC head renaming
    assert _keys_are_equivalent("model.last_linear.1.weight", "fc.weight")
    assert _keys_are_equivalent("model.last_linear.1.bias", "fc.bias")

    # Non-matches
    assert not _keys_are_equivalent("conv1.weight", "conv2.weight")
    assert not _keys_are_equivalent("model.conv1.weight", "conv2.weight")
    assert not _keys_are_equivalent("random_key", "conv1.weight")
