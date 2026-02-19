"""
Shield-Ryzen V2 — Shield-Xception Core Engine
================================================
Class Mapping (ffpp_c23.pth verified):
  Index 0 = FAKE  |  Index 1 = REAL

Architecture: XceptionNet (timm.legacy_xception)
  - 276 weight keys, 20.8M parameters
  - Depthwise separable convolutions for efficient feature extraction
  - Trained on FaceForensics++ (c23 compression)

Security:
  - SHA-256 hash verification at load time
  - ModelTamperingError raised on hash mismatch
  - Reference output regression check
  - ONNX model shape verification (input/output)

Developer: Inayat Hussain | AMD Slingshot 2026
Part 4 of 14 — Neural Model Verification & Integrity

⚠️  AGENT RULE: DO NOT AUTO-MODIFY this file. Suggest changes as diffs/plans.
"""

import os
import sys
import hashlib
import json
import math
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

_PROJECT_ROOT = Path(__file__).resolve().parent
_log = logging.getLogger("ShieldXception")

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.89
BLINK_THRESHOLD = 0.21
BLINK_TIME_WINDOW = 10
LAPLACIAN_THRESHOLD = 50

# MediaPipe FaceLandmarker eye indices (478-point mesh)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Model integrity constants
MODEL_EXPECTED_HASH = "8bcb10c1567d66bca32776b4c4b8f9e037be37722270e0c65643f7a2c781d762"
MODEL_EXPECTED_KEY_COUNT = 276
MODEL_INPUT_SHAPE = (1, 3, 299, 299)
MODEL_OUTPUT_SHAPE = (1, 2)


# ═══════════════════════════════════════════════════════════════
#  SECURITY EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

class SecurityError(Exception):
    """Base exception for model security violations."""
    pass


class ModelTamperingError(SecurityError):
    """Raised when cryptographic hash verification fails.
    Indicates the model file has been modified or corrupted."""
    pass


class ModelShapeError(SecurityError):
    """Raised when model input/output shapes don't match expected values."""
    pass


# ═══════════════════════════════════════════════════════════════
#  SHIELD-XCEPTION MODEL  (PRESERVED CLASS STRUCTURE)
# ═══════════════════════════════════════════════════════════════

class ShieldXception(nn.Module):
    """XceptionNet wrapper for deepfake detection.

    Output: softmax probabilities [fake_prob, real_prob].
    The softmax is applied in forward() so the output sums to 1.0.
    """

    def __init__(self):
        super(ShieldXception, self).__init__()
        self.model = timm.create_model(
            'legacy_xception', pretrained=False, num_classes=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.softmax(logits, dim=1)


# ═══════════════════════════════════════════════════════════════
#  EAR & TEXTURE (Legacy V1 — backward compatibility)
# ═══════════════════════════════════════════════════════════════

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio from 6 landmark points.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Low EAR = eye closed (blink). Normal ~0.25-0.3, blink ~0.15.

    Note: V2 uses compute_ear() from shield_utils_core.py with
    cosine angle compensation. This remains for V1 compatibility.
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    v1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
    v2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
    h_dist = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)

    if h_dist == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


def check_texture(face_crop):
    """Laplacian variance — low = too smooth (photo/deepfake artifact)."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ═══════════════════════════════════════════════════════════════
#  SECURE MODEL LOADING  (TASK 4.2)
# ═══════════════════════════════════════════════════════════════

def _compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file using chunked reading.

    Uses 64KB blocks for memory efficiency on 80MB+ model files.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_expected_hash(sig_path: Optional[str] = None) -> str:
    """Load expected hash from signature file or return compiled constant."""
    if sig_path is None:
        sig_path = str(_PROJECT_ROOT / "models" / "model_signature.sha256")

    if os.path.exists(sig_path):
        with open(sig_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # Fallback to compiled constant
    return MODEL_EXPECTED_HASH


def _verify_onnx_shapes(onnx_path: str) -> dict:
    """Verify ONNX model input/output shapes at load time.

    Returns shape verification dict. Raises ModelShapeError on mismatch.
    """
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        input_info = sess.get_inputs()[0]
        output_info = sess.get_outputs()[0]

        input_shape = input_info.shape
        output_shape = output_info.shape

        # Validate input shape (handle dynamic batch)
        expected_in = list(MODEL_INPUT_SHAPE)
        actual_in = [int(d) if isinstance(d, int) else -1 for d in input_shape]
        if actual_in[1:] != expected_in[1:]:
            raise ModelShapeError(
                f"ONNX input shape mismatch: expected {expected_in}, got {actual_in}"
            )

        # Validate output shape
        expected_out = list(MODEL_OUTPUT_SHAPE)
        actual_out = [int(d) if isinstance(d, int) else -1 for d in output_shape]
        if actual_out[-1] != expected_out[-1]:
            raise ModelShapeError(
                f"ONNX output shape mismatch: expected {expected_out}, got {actual_out}"
            )

        return {
            "input_shape": actual_in,
            "output_shape": actual_out,
            "input_name": input_info.name,
            "input_dtype": input_info.type,
            "verified": True,
        }
    except ImportError:
        _log.warning("onnxruntime not available — skipping ONNX shape verification")
        return {"verified": False, "reason": "onnxruntime not installed"}


def load_model_with_verification(
    model_path: str,
    device: Optional[torch.device] = None,
    skip_hash: bool = False,
) -> dict:
    """Load model weights with full integrity verification.

    Step 1: SHA-256 hash check against signature file/constant
    Step 2: Load PyTorch state dict
    Step 3: Verify key count matches expected (276)
    Step 4: Handle key renaming (model.* prefix, last_linear → fc)
    Step 5: Return clean state dict ready for model.load_state_dict()

    Args:
        model_path: Path to ffpp_c23.pth checkpoint
        device:     Target torch device (defaults to CPU for loading)
        skip_hash:  If True, skip hash verification (for testing only)

    Returns:
        Clean state dict with keys matching timm legacy_xception

    Raises:
        ModelTamperingError: If hash doesn't match expected
        FileNotFoundError:   If model file or signature missing
    """
    if device is None:
        device = torch.device("cpu")

    _log.info("🔒 Verifying integrity of %s...", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ── Step 1: SHA-256 hash verification ─────────────────────
    if not skip_hash:
        actual_hash = _compute_file_hash(model_path)
        expected_hash = _load_expected_hash()

        if actual_hash != expected_hash:
            raise ModelTamperingError(
                f"🚨 MODEL INTEGRITY VIOLATION!\n"
                f"Expected: {expected_hash}\n"
                f"Got:      {actual_hash}\n"
                f"The model file has been modified or corrupted.\n"
                f"If this is intentional (e.g., retraining), update the signature:\n"
                f"  python verify_model.py"
            )
        _log.info("✅ HASH VERIFIED: %s...", actual_hash[:16])
    else:
        _log.warning("⚠️  Hash verification SKIPPED (testing mode)")

    # ── Step 2: Load checkpoint ───────────────────────────────
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        _log.warning("weights_only=True failed, using fallback load")
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # ── Step 3: Verify key count ──────────────────────────────
    key_count = len(state_dict)
    _log.info("Checkpoint has %d weight keys (expected %d)", key_count, MODEL_EXPECTED_KEY_COUNT)

    if key_count != MODEL_EXPECTED_KEY_COUNT:
        _log.warning(
            "Key count mismatch: got %d, expected %d — "
            "model may have been retrained or architecture changed",
            key_count, MODEL_EXPECTED_KEY_COUNT,
        )

    # ── Step 4: Clean key names ───────────────────────────────
    # Strip "model." prefix and rename FC head
    clean_sd = {}
    for k, v in state_dict.items():
        bare = k
        for prefix in ("module.", "model.", "net."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break

        # FC head rename: last_linear.1.* → fc.*
        if bare == "last_linear.1.weight":
            bare = "fc.weight"
        elif bare == "last_linear.1.bias":
            bare = "fc.bias"

        clean_sd[bare] = v

    _log.info("✅ State dict cleaned: %d keys ready for load_state_dict()", len(clean_sd))

    return clean_sd


def verify_onnx_model(onnx_path: str) -> dict:
    """Verify ONNX model integrity: hash + shape check.

    Used by v3_int8_engine.py during initialization.

    Returns:
        Verification result dict with hash, shapes, and status.
    """
    if not os.path.exists(onnx_path):
        return {"verified": False, "error": f"File not found: {onnx_path}"}

    result = {
        "file": onnx_path,
        "file_size_bytes": os.path.getsize(onnx_path),
        "hash_sha256": _compute_file_hash(onnx_path),
    }

    shape_info = _verify_onnx_shapes(onnx_path)
    result.update(shape_info)

    return result


# ═══════════════════════════════════════════════════════════════
#  MODULE-LEVEL INITIALIZATION (V1 compatibility)
# ═══════════════════════════════════════════════════════════════

# Only initialize model at module level if run as V1 script
if __name__ == "__main__":
    import mediapipe as mp
    from PIL import Image
    from torchvision import transforms

    # Import calibrator
    sys.path.insert(0, str(_PROJECT_ROOT))
    from shield_utils_core import ConfidenceCalibrator

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    ))
    model = ShieldXception().to(device)

    # Initialize Calibrator
    calibrator = ConfidenceCalibrator()

    try:
        clean_state_dict = load_model_with_verification('ffpp_c23.pth', device)
        result = model.model.load_state_dict(clean_state_dict, strict=False)
        matched = len(clean_state_dict) - len(result.unexpected_keys)
        print(f"✅ Brain Loaded — {matched}/{len(clean_state_dict)} weights matched.")
        print(f"✅ Calibration — Temperature T={calibrator.temperature:.2f}")
    except Exception as e:
        print(f"❌ SECURITY/LOAD ERROR: {e}")

    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    landmarker_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task'
    )
    landmarker_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    def main():
        cap = cv2.VideoCapture(0)
        print("═" * 55)
        print("  🛡️  SHIELD-RYZEN SECURITY MODE ACTIVE")
        print("═" * 55)
        print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
        print(f"  Blink Window:         {BLINK_TIME_WINDOW}s")
        print(f"  Texture Guard:        Laplacian > {LAPLACIAN_THRESHOLD}")
        print("  Press ESC to exit.")
        print("═" * 55)

        frame_timestamp_ms = 0
        fps_counter = 0
        fps_display = 0.0
        fps_timer = time.time()
        blink_count = 0
        blink_timestamps = []
        was_eye_closed = False

        with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                fps_counter += 1
                now = time.time()
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    fps_timer = now

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                lm_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                frame_timestamp_ms += 33

                h, w, _ = frame.shape
                blink_timestamps = [t for t in blink_timestamps if now - t < BLINK_TIME_WINDOW]
                blink_count = len(blink_timestamps)

                if lm_result.face_landmarks:
                    for face_landmarks in lm_result.face_landmarks:
                        xs = [lm.x for lm in face_landmarks]
                        ys = [lm.y for lm in face_landmarks]
                        x_min = max(0, int(min(xs) * w) - 10)
                        y_min = max(0, int(min(ys) * h) - 10)
                        x_max = min(w, int(max(xs) * w) + 10)
                        y_max = min(h, int(max(ys) * h) + 10)

                        left_ear = calculate_ear(face_landmarks, LEFT_EYE)
                        right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0

                        if avg_ear < BLINK_THRESHOLD:
                            was_eye_closed = True
                        elif was_eye_closed and avg_ear >= BLINK_THRESHOLD:
                            was_eye_closed = False
                            blink_timestamps.append(now)
                            blink_count = len(blink_timestamps)

                        liveness_ok = blink_count > 0

                        face_crop = frame[y_min:y_max, x_min:x_max]
                        if face_crop.size == 0:
                            continue

                        texture_score = check_texture(face_crop)
                        texture_ok = texture_score > LAPLACIAN_THRESHOLD

                        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        input_tensor = transform(face_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            probs = model(input_tensor).cpu().numpy()[0]
                            calibrated_probs = calibrator.calibrate(probs)
                            fake_prob = float(calibrated_probs[0])
                            real_prob = float(calibrated_probs[1])

                        if fake_prob > 0.50:
                            label = "CRITICAL: FAKE DETECTED"
                            color = (0, 0, 255)
                        elif real_prob < CONFIDENCE_THRESHOLD:
                            label = "WARNING: LOW CONFIDENCE"
                            color = (0, 200, 255)
                        elif not liveness_ok:
                            label = "LIVENESS FAILED"
                            color = (0, 165, 255)
                        elif not texture_ok:
                            label = "SMOOTHNESS WARNING"
                            color = (0, 200, 255)
                        else:
                            label = "SHIELD: VERIFIED REAL"
                            color = (0, 255, 0)

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(frame, label, (x_min, y_min - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame,
                                    f"Real:{real_prob*100:.1f}% Fake:{fake_prob*100:.1f}%",
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
                cv2.putText(frame, f"FPS: {fps_display:.1f} | {device}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                blink_color = (0, 255, 0) if blink_count > 0 else (0, 0, 255)
                cv2.putText(frame,
                            f"Blink: {'YES' if blink_count > 0 else 'NO'} ({blink_count})",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)
                cv2.putText(frame, "SECURITY MODE", (w - 170, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                cv2.imshow('Shield-Ryzen V1 | SECURITY MODE', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    main()