"""
Shield-Ryzen V2 — Emergency Diagnostic Script
===============================================
THE LOGIC COLLAPSE DEBUGGER — Identifies root causes for:
  - Blink detection not counting
  - AI deepfakes declared "100% Human"
  - Broken normalization / preprocessing

Run this IMMEDIATELY to find failures before applying fixes.

Diagnostics:
  1. Normalization Verification
  2. Landmark Index Verification
  3. EAR Value Stream (live 10-second capture)
  4. Model Output Analysis (inference sanity check)
  5. Tier Isolation (which tier causes false positives)

Usage:
    python scripts/emergency_diagnostic.py [--no-camera] [--duration 10]

Output:
    diagnostics/logic_collapse_report.json
    diagnostics/landmark_debug.png

Developer: Inayat Hussain | AMD Slingshot 2026
Part 1 of 14 — Foundation Diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ─── Output Directory ────────────────────────────────────────
_DIAG_DIR = os.path.join(_project_root, "diagnostics")
os.makedirs(_DIAG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Normalization Verification
# ═══════════════════════════════════════════════════════════════

def _diag_normalization(frame: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Verify that preprocessing matches XceptionNet training normalization.

    Checks:
      - Pixel stats BEFORE preprocessing (raw BGR uint8)
      - Pixel stats AFTER preprocessing (normalized float32 NCHW)
      - Whether output matches [-1, 1] range (FF++ standard)
      - Whether output matches ImageNet range (~[-2, 2])
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 1: Normalization Verification")
    print("=" * 60)

    result: dict[str, Any] = {"diagnostic": "normalization", "passed": False, "flags": []}

    try:
        from shield_utils_core import preprocess_face
    except ImportError as e:
        result["error"] = f"Cannot import preprocess_face: {e}"
        result["flags"].append("IMPORT_FAILURE: shield_utils_core.preprocess_face not found")
        return result

    # Use provided frame or create synthetic one
    if frame is None:
        frame = np.random.RandomState(42).randint(40, 220, (480, 640, 3), dtype=np.uint8)
        result["input_source"] = "synthetic"
    else:
        result["input_source"] = "live_camera"

    # Before preprocessing
    before = {
        "min": int(frame.min()),
        "max": int(frame.max()),
        "mean": round(float(frame.mean()), 4),
        "std": round(float(frame.std()), 4),
        "dtype": str(frame.dtype),
        "shape": list(frame.shape),
    }
    result["before_preprocessing"] = before
    print(f"  BEFORE: min={before['min']} max={before['max']} mean={before['mean']:.2f} std={before['std']:.2f}")

    # After preprocessing
    try:
        tensor = preprocess_face(frame)
    except Exception as e:
        result["error"] = f"preprocess_face crashed: {e}"
        result["flags"].append(f"CRASH: preprocess_face exception — {e}")
        return result

    after = {
        "min": round(float(tensor.min()), 6),
        "max": round(float(tensor.max()), 6),
        "mean": round(float(tensor.mean()), 6),
        "std": round(float(tensor.std()), 6),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    result["after_preprocessing"] = after
    print(f"  AFTER:  min={after['min']:.4f} max={after['max']:.4f} mean={after['mean']:.4f} std={after['std']:.4f}")

    # Per-channel stats
    if tensor.ndim == 4 and tensor.shape[1] == 3:
        for c, name in enumerate(["R/B", "G", "B/R"]):
            ch = tensor[0, c, :, :]
            print(f"  Channel {c} ({name}): mean={float(ch.mean()):.4f} std={float(ch.std()):.4f}")

    # Option A: [-1, 1] normalization (FF++ standard)
    ff_match = (-1.05 <= after["min"] <= -0.5) and (0.5 <= after["max"] <= 1.05)
    # Option B: ImageNet normalization
    imagenet_match = (-3.0 <= after["min"] <= -1.5) and (1.5 <= after["max"] <= 3.0)
    # Broken: still in [0, 255] range
    raw_range = after["min"] >= 0 and after["max"] > 1.5

    if ff_match:
        result["normalization_type"] = "FF++ [-1, 1]"
        result["match"] = "OPTION_B_FF"
        result["passed"] = True
        print("  ✅ MATCH: Preprocessing uses FF++ [-1,1] normalization (CORRECT)")
    elif imagenet_match:
        result["normalization_type"] = "ImageNet"
        result["match"] = "OPTION_A_IMAGENET"
        result["passed"] = False
        result["flags"].append("MISMATCH: Using ImageNet normalization but model expects [-1,1]")
        print("  ⚠️  MISMATCH: Preprocessing uses ImageNet normalization, but FF++ expects [-1,1]")
    elif raw_range:
        result["normalization_type"] = "NONE (raw pixel values)"
        result["match"] = "NO_NORMALIZATION"
        result["passed"] = False
        result["flags"].append("CRITICAL: No normalization applied! Model receiving raw [0,255] values")
        print("  ❌ MISMATCH: Expected range [-1,1] but got [{:.2f}, {:.2f}]".format(
            after["min"], after["max"]))
        print("     THIS IS WHY YOUR MODEL OUTPUTS GARBAGE.")
    else:
        result["normalization_type"] = "Unknown"
        result["match"] = "UNKNOWN"
        result["passed"] = False
        result["flags"].append(f"UNKNOWN range: [{after['min']:.4f}, {after['max']:.4f}]")
        print(f"  ⚠️  Unknown normalization range: [{after['min']:.4f}, {after['max']:.4f}]")

    # Verify formula: (128/255 - 0.5) / 0.5 ≈ 0.00392
    test_crop = np.full((100, 100, 3), 128, dtype=np.uint8)
    test_tensor = preprocess_face(test_crop)
    neutral_val = float(test_tensor.mean())
    expected_neutral = (128.0 / 255.0 - 0.5) / 0.5  # ≈ 0.00392
    result["neutral_point_test"] = {
        "expected": round(expected_neutral, 6),
        "actual": round(neutral_val, 6),
        "delta": round(abs(neutral_val - expected_neutral), 6),
    }
    if abs(neutral_val - expected_neutral) < 0.02:
        print(f"  ✅ Neutral point: expected={expected_neutral:.4f} actual={neutral_val:.4f} — OK")
    else:
        print(f"  ❌ Neutral point drift: expected={expected_neutral:.4f} got={neutral_val:.4f}")
        result["flags"].append(f"Neutral point drift: {neutral_val:.4f} vs expected {expected_neutral:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Landmark Index Verification
# ═══════════════════════════════════════════════════════════════

def _diag_landmarks(frame: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Verify MediaPipe 478-point landmark eye indices are correct.

    Extracts eye landmarks and draws them on a debug image to confirm
    they correspond to actual eye positions.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 2: Landmark Index Verification")
    print("=" * 60)

    result: dict[str, Any] = {"diagnostic": "landmarks", "passed": False, "flags": []}

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    if frame is None:
        result["skipped"] = True
        result["reason"] = "No camera frame available; cannot test landmarks"
        result["flags"].append("SKIPPED: No live frame for landmark detection")
        print("  ⚠️  Skipped — no live frame available (use --no-camera to skip)")
        # Still document expected indices
        result["expected_indices"] = {
            "left_eye": LEFT_EYE,
            "right_eye": RIGHT_EYE,
            "description": "MediaPipe 478-point FaceMesh v2 indices"
        }
        return result

    try:
        from shield_face_pipeline import ShieldFacePipeline
    except ImportError:
        try:
            import mediapipe as mp
        except ImportError:
            result["error"] = "Neither ShieldFacePipeline nor mediapipe available"
            return result

    # Try using ShieldFacePipeline for landmark extraction
    try:
        pipeline = ShieldFacePipeline(detector_type="mediapipe", min_detection_confidence=0.5)
        faces = pipeline.detect_faces(frame)

        if not faces:
            result["flags"].append("NO_FACE_DETECTED: Cannot verify landmarks without a face")
            print("  ⚠️  No face detected in frame — aim the camera at a face")
            result["no_face"] = True
            pipeline.release()
            return result

        face = faces[0]
        landmarks = face.landmarks

        if landmarks is None:
            result["flags"].append("NO_LANDMARKS: Face detected but no landmarks returned")
            print("  ❌ Face detected but no landmarks returned")
            pipeline.release()
            return result

        # Extract eye landmark coordinates
        h, w = frame.shape[:2]
        left_coords = []
        right_coords = []

        for idx in LEFT_EYE:
            if idx < len(landmarks):
                lm = landmarks[idx]
                # MediaPipe landmarks can be (x, y, z) or objects with .x, .y
                if hasattr(lm, 'x'):
                    px, py = int(lm.x * w), int(lm.y * h)
                else:
                    px, py = int(lm[0] * w), int(lm[1] * h)
                left_coords.append((px, py))
            else:
                left_coords.append(None)
                result["flags"].append(f"LEFT_EYE index {idx} out of range (total: {len(landmarks)})")

        for idx in RIGHT_EYE:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if hasattr(lm, 'x'):
                    px, py = int(lm.x * w), int(lm.y * h)
                else:
                    px, py = int(lm[0] * w), int(lm[1] * h)
                right_coords.append((px, py))
            else:
                right_coords.append(None)
                result["flags"].append(f"RIGHT_EYE index {idx} out of range (total: {len(landmarks)})")

        result["left_eye_coords"] = [list(c) if c else None for c in left_coords]
        result["right_eye_coords"] = [list(c) if c else None for c in right_coords]
        result["total_landmarks"] = len(landmarks)

        print(f"  Total landmarks: {len(landmarks)}")
        for i, (idx, coord) in enumerate(zip(LEFT_EYE, left_coords)):
            print(f"  LEFT_EYE[{i}] (idx {idx}): {coord}")
        for i, (idx, coord) in enumerate(zip(RIGHT_EYE, right_coords)):
            print(f"  RIGHT_EYE[{i}] (idx {idx}): {coord}")

        # Sanity check: left eye should be on left side, right eye on right
        valid_left = [c for c in left_coords if c is not None]
        valid_right = [c for c in right_coords if c is not None]

        if valid_left and valid_right:
            left_center_x = np.mean([c[0] for c in valid_left])
            right_center_x = np.mean([c[0] for c in valid_right])

            # In a mirrored webcam, "left eye" appears on right side of image
            # Check that they are separated horizontally
            separation = abs(right_center_x - left_center_x)
            if separation < 10:
                result["flags"].append("WARNING: Eyes appear at same X position — landmark indices may be wrong")
                print("  ⚠️  WARNING: Eyes overlap horizontally — check indices")
            else:
                print(f"  ✅ Eye separation: {separation:.0f}px (looks correct)")
                result["passed"] = True

            # Check that eyes are roughly at the same Y level
            left_center_y = np.mean([c[1] for c in valid_left])
            right_center_y = np.mean([c[1] for c in valid_right])
            y_diff = abs(left_center_y - right_center_y)
            if y_diff > 50:
                result["flags"].append(f"WARNING: Eyes at different Y levels (diff={y_diff:.0f}px)")
                print(f"  ⚠️  Y-level difference: {y_diff:.0f}px (head may be tilted)")

        # Draw debug image
        debug_img = frame.copy()
        for coord in valid_left:
            cv2.circle(debug_img, coord, 3, (0, 255, 0), -1)  # Green for left
        for coord in valid_right:
            cv2.circle(debug_img, coord, 3, (0, 0, 255), -1)  # Red for right

        # Draw connecting lines
        if len(valid_left) >= 2:
            for i in range(len(valid_left) - 1):
                cv2.line(debug_img, valid_left[i], valid_left[i + 1], (0, 255, 0), 1)
            cv2.line(debug_img, valid_left[-1], valid_left[0], (0, 255, 0), 1)
        if len(valid_right) >= 2:
            for i in range(len(valid_right) - 1):
                cv2.line(debug_img, valid_right[i], valid_right[i + 1], (0, 0, 255), 1)
            cv2.line(debug_img, valid_right[-1], valid_right[0], (0, 0, 255), 1)

        debug_path = os.path.join(_DIAG_DIR, "landmark_debug.png")
        cv2.imwrite(debug_path, debug_img)
        print(f"  📸 Debug image saved → {debug_path}")
        result["debug_image"] = debug_path

        pipeline.release()

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["flags"].append(f"EXCEPTION: {e}")
        print(f"  ❌ Error: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: EAR Value Stream
# ═══════════════════════════════════════════════════════════════

def _diag_ear_stream(
    camera_id: int = 0,
    duration_seconds: int = 10,
    use_camera: bool = True,
) -> dict[str, Any]:
    """Capture EAR values over time to verify blink detection is possible.

    Computes EAR for N frames and checks if EAR drops are detectable.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 3: EAR Value Stream")
    print("=" * 60)

    result: dict[str, Any] = {"diagnostic": "ear_stream", "passed": False, "flags": []}

    if not use_camera:
        result["skipped"] = True
        result["reason"] = "Camera disabled (--no-camera)"
        print("  ⚠️  Skipped — camera disabled")
        return result

    try:
        from shield_camera import ShieldCamera
        from shield_face_pipeline import ShieldFacePipeline
        from shield_utils_core import compute_ear
    except ImportError as e:
        result["error"] = f"Import error: {e}"
        return result

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    cam = ShieldCamera(camera_id=camera_id)
    pipeline = ShieldFacePipeline(detector_type="mediapipe", min_detection_confidence=0.5)

    ear_values: list[float] = []
    timestamps: list[float] = []
    frames_processed = 0
    frames_no_face = 0

    print(f"  Capturing EAR values for {duration_seconds} seconds...")
    print("  👁️  Blink deliberately 5 times during this capture!")
    print()

    start = time.monotonic()
    while (time.monotonic() - start) < duration_seconds:
        ok, frame, ts = cam.read_validated_frame()
        if not ok:
            continue

        faces = pipeline.detect_faces(frame)
        if not faces:
            frames_no_face += 1
            continue

        face = faces[0]
        if face.landmarks is None:
            frames_no_face += 1
            continue

        ear_l, rel_l = compute_ear(face.landmarks, LEFT_EYE, face.head_pose, face.is_frontal)
        ear_r, rel_r = compute_ear(face.landmarks, RIGHT_EYE, face.head_pose, face.is_frontal)
        ear = (ear_l + ear_r) / 2.0

        ear_values.append(ear)
        timestamps.append(time.monotonic() - start)
        frames_processed += 1

        # Print live EAR value every 10th frame
        if frames_processed % 10 == 0:
            bar = "█" * int(ear * 50)
            print(f"  EAR: {ear:.4f} {bar}")

    cam.release()
    pipeline.release()

    result["frames_processed"] = frames_processed
    result["frames_no_face"] = frames_no_face
    result["duration_actual"] = round(time.monotonic() - start, 2)

    if not ear_values:
        result["flags"].append("NO_EAR_DATA: No faces detected during capture")
        print("  ❌ No EAR data collected — no faces detected!")
        return result

    ear_arr = np.array(ear_values)
    stats = {
        "min": round(float(ear_arr.min()), 6),
        "max": round(float(ear_arr.max()), 6),
        "mean": round(float(ear_arr.mean()), 6),
        "std": round(float(ear_arr.std()), 6),
        "range": round(float(ear_arr.max() - ear_arr.min()), 6),
    }
    result["ear_stats"] = stats

    print(f"\n  EAR Statistics:")
    print(f"    Min:   {stats['min']:.4f}")
    print(f"    Max:   {stats['max']:.4f}")
    print(f"    Mean:  {stats['mean']:.4f}")
    print(f"    Std:   {stats['std']:.4f}")
    print(f"    Range: {stats['range']:.4f}")

    # Check for blink-like dips
    BLINK_THRESH = 0.21
    below_thresh = (ear_arr < BLINK_THRESH).sum()
    result["samples_below_threshold"] = int(below_thresh)

    # Analyze EAR variability
    if stats["std"] < 0.005:
        result["flags"].append(
            f"CONSTANT_EAR: EAR std={stats['std']:.6f} — values barely change. "
            "Landmarks may be wrong or face is too far."
        )
        print(f"  ❌ FLAG: EAR is nearly constant (std={stats['std']:.6f}) — landmarks may be broken")
    elif stats["range"] < 0.03:
        result["flags"].append(
            f"LOW_RANGE: EAR range={stats['range']:.4f} — may not detect blinks"
        )
        print(f"  ⚠️  FLAG: EAR range too small ({stats['range']:.4f}) — blinks may be undetectable")
    else:
        print(f"  ✅ EAR shows variation — blinks should be detectable")

    # Check threshold vs mean
    if stats["mean"] < BLINK_THRESH:
        result["flags"].append(
            f"ALWAYS_BELOW: Open-eye EAR ({stats['mean']:.4f}) is below threshold ({BLINK_THRESH}). "
            "System thinks eyes are ALWAYS closed = always 'blinking'. "
            "Threshold needs to be lowered."
        )
        print(f"  ❌ FLAG: Average EAR ({stats['mean']:.4f}) is BELOW threshold ({BLINK_THRESH})")
        print(f"     → System thinks eyes are ALWAYS closed!")
    elif stats["mean"] - BLINK_THRESH < 0.02:
        result["flags"].append(
            f"MARGINAL: Open-eye EAR ({stats['mean']:.4f}) is barely above threshold ({BLINK_THRESH}). "
            "Noise alone could trigger false blinks."
        )
        print(f"  ⚠️  FLAG: EAR ({stats['mean']:.4f}) barely above threshold ({BLINK_THRESH})")

    result["passed"] = len(result["flags"]) == 0

    return result


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Model Output Analysis
# ═══════════════════════════════════════════════════════════════

def _diag_model_output(
    camera_id: int = 0,
    n_frames: int = 100,
    use_camera: bool = True,
) -> dict[str, Any]:
    """Analyze model inference outputs for anomalies.

    Checks:
      - Whether outputs are stuck (always same value = broken)
      - Whether confidence is always >95% (uncalibrated)
      - Compares FP32 vs INT8 models if both exist
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 4: Model Output Analysis")
    print("=" * 60)

    result: dict[str, Any] = {"diagnostic": "model_output", "passed": False, "flags": []}

    try:
        import onnxruntime as ort
    except ImportError:
        result["error"] = "onnxruntime not installed"
        return result

    try:
        from shield_utils_core import preprocess_face
    except ImportError as e:
        result["error"] = f"Import error: {e}"
        return result

    # Locate models
    int8_path = os.path.join(_project_root, "shield_ryzen_int8.onnx")
    fp32_path = os.path.join(_project_root, "shield_ryzen_v2.onnx")

    models: dict[str, str] = {}
    if os.path.exists(int8_path):
        models["INT8"] = int8_path
    if os.path.exists(fp32_path):
        models["FP32"] = fp32_path

    if not models:
        result["flags"].append("NO_MODELS: Neither INT8 nor FP32 ONNX model found")
        print("  ❌ No ONNX models found!")
        return result

    # Get test frames
    test_crops: list[np.ndarray] = []

    if use_camera:
        try:
            from shield_camera import ShieldCamera
            from shield_face_pipeline import ShieldFacePipeline

            cam = ShieldCamera(camera_id=camera_id)
            pipeline = ShieldFacePipeline(detector_type="mediapipe", min_detection_confidence=0.5)

            collected = 0
            attempts = 0
            while collected < min(n_frames, 50) and attempts < 200:
                ok, frame, ts = cam.read_validated_frame()
                if ok:
                    faces = pipeline.detect_faces(frame)
                    if faces and faces[0].face_crop_raw is not None:
                        test_crops.append(faces[0].face_crop_raw)
                        collected += 1
                attempts += 1

            cam.release()
            pipeline.release()
            print(f"  Collected {len(test_crops)} live face crops")
        except Exception as e:
            print(f"  ⚠️  Camera capture failed: {e}")

    # Add synthetic crops if needed
    if len(test_crops) < 10:
        rng = np.random.RandomState(42)
        for _ in range(max(10, n_frames - len(test_crops))):
            synthetic = rng.randint(40, 220, (299, 299, 3), dtype=np.uint8)
            test_crops.append(synthetic)
        result["used_synthetic_crops"] = True
        print(f"  Added synthetic crops (total: {len(test_crops)})")

    # Test each model
    for model_name, model_path in models.items():
        print(f"\n  Testing {model_name} model: {os.path.basename(model_path)}")

        try:
            sess = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            input_name = sess.get_inputs()[0].name
            active_provider = sess.get_providers()[0]
            print(f"    Provider: {active_provider}")

            outputs: list[list[float]] = []
            for crop in test_crops[:n_frames]:
                tensor = preprocess_face(crop)
                raw = sess.run(None, {input_name: tensor})[0][0]
                outputs.append([float(x) for x in raw])

            out_arr = np.array(outputs)
            model_result: dict[str, Any] = {
                "provider": active_provider,
                "n_samples": len(outputs),
                "output_shape": list(out_arr.shape),
            }

            # Per-class stats
            for i, cls_name in enumerate(["real", "fake"]):
                if i < out_arr.shape[1]:
                    col = out_arr[:, i]
                    model_result[f"{cls_name}_prob"] = {
                        "min": round(float(col.min()), 6),
                        "max": round(float(col.max()), 6),
                        "mean": round(float(col.mean()), 6),
                        "std": round(float(col.std()), 6),
                    }

            # Check if stuck
            if out_arr.shape[1] >= 2:
                real_std = float(out_arr[:, 0].std())
                fake_std = float(out_arr[:, 1].std())

                if real_std < 0.001 and fake_std < 0.001:
                    model_result["stuck"] = True
                    result["flags"].append(
                        f"{model_name}_STUCK: Model outputs are constant "
                        f"(real_std={real_std:.6f}, fake_std={fake_std:.6f}). "
                        "Model is broken — likely preprocessing or quantization damage."
                    )
                    print(f"    ❌ STUCK: Outputs barely change (std < 0.001)")
                else:
                    model_result["stuck"] = False

                # Check overconfidence
                max_conf = float(out_arr.max(axis=1).mean())
                if max_conf > 0.95:
                    result["flags"].append(
                        f"{model_name}_OVERCONFIDENT: Average max confidence = {max_conf:.4f} (>95%). "
                        "Model is uncalibrated."
                    )
                    print(f"    ⚠️  Overconfident: avg max confidence = {max_conf:.4f}")

                # Print sample outputs
                print(f"    Sample outputs (first 5):")
                for j in range(min(5, len(outputs))):
                    print(f"      [{outputs[j][0]:.4f}, {outputs[j][1]:.4f}]")

            result[model_name] = model_result

        except Exception as e:
            result[model_name] = {"error": str(e)}
            result["flags"].append(f"{model_name}_LOAD_FAILED: {e}")
            print(f"    ❌ Error: {e}")

    # Compare FP32 vs INT8
    if "FP32" in result and "INT8" in result:
        fp32_ok = not result["FP32"].get("stuck", True)
        int8_ok = not result["INT8"].get("stuck", True)

        if fp32_ok and not int8_ok:
            result["flags"].append(
                "QUANTIZATION_DAMAGE: FP32 model works but INT8 is stuck. "
                "INT8 quantization has destroyed the model's discriminative ability."
            )
            print("  ❌ QUANTIZATION DAMAGE: FP32 works but INT8 is broken!")
        elif not fp32_ok and not int8_ok:
            result["flags"].append(
                "PREPROCESSING_BUG: Both models produce stuck outputs. "
                "The problem is likely in normalization/preprocessing, NOT the model."
            )
            print("  ❌ PREPROCESSING BUG: Both models broken — check normalization!")

    result["passed"] = len(result["flags"]) == 0

    return result


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: Tier Isolation
# ═══════════════════════════════════════════════════════════════

def _diag_tier_isolation(
    camera_id: int = 0,
    n_frames: int = 30,
    use_camera: bool = True,
) -> dict[str, Any]:
    """Run each detection tier independently and log results.

    Identifies WHICH tier is causing false positives by testing:
      Tier 1 (Neural):   What does the model actually output?
      Tier 2 (Liveness): What are the actual EAR values?
      Tier 3 (Forensic): What is the actual Laplacian variance?
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 5: Tier Isolation")
    print("=" * 60)

    result: dict[str, Any] = {"diagnostic": "tier_isolation", "passed": False, "flags": []}

    if not use_camera:
        result["skipped"] = True
        result["reason"] = "Camera disabled"
        print("  ⚠️  Skipped — camera disabled")
        return result

    try:
        from shield_camera import ShieldCamera
        from shield_face_pipeline import ShieldFacePipeline
        from shield_utils_core import compute_ear, compute_texture_score, preprocess_face
        import onnxruntime as ort
    except ImportError as e:
        result["error"] = f"Import error: {e}"
        return result

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Load model
    int8_path = os.path.join(_project_root, "shield_ryzen_int8.onnx")
    model_path = int8_path if os.path.exists(int8_path) else os.path.join(_project_root, "shield_ryzen_v2.onnx")

    sess = None
    input_name = None
    if os.path.exists(model_path):
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            input_name = sess.get_inputs()[0].name
        except Exception as e:
            result["flags"].append(f"MODEL_LOAD_FAILED: {e}")

    cam = ShieldCamera(camera_id=camera_id)
    pipeline = ShieldFacePipeline(detector_type="mediapipe", min_detection_confidence=0.5)

    tier1_verdicts: list[str] = []
    tier2_verdicts: list[str] = []
    tier3_verdicts: list[str] = []

    neural_confs: list[float] = []
    ear_vals: list[float] = []
    texture_vals: list[float] = []

    collected = 0
    attempts = 0

    while collected < n_frames and attempts < n_frames * 5:
        ok, frame, ts = cam.read_validated_frame()
        if not ok:
            attempts += 1
            continue

        faces = pipeline.detect_faces(frame)
        if not faces:
            attempts += 1
            continue

        face = faces[0]

        # Tier 1: Neural
        if sess and face.face_crop_raw is not None:
            tensor = preprocess_face(face.face_crop_raw)
            raw = sess.run(None, {input_name: tensor})[0][0]
            real_prob = float(raw[1]) if len(raw) > 1 else float(raw[0])
            verdict = "REAL" if real_prob > 0.5 else "FAKE"
            tier1_verdicts.append(verdict)
            neural_confs.append(real_prob)
        else:
            tier1_verdicts.append("N/A")

        # Tier 2: Liveness (EAR)
        if face.landmarks is not None:
            ear_l, _ = compute_ear(face.landmarks, LEFT_EYE, face.head_pose, face.is_frontal)
            ear_r, _ = compute_ear(face.landmarks, RIGHT_EYE, face.head_pose, face.is_frontal)
            ear = (ear_l + ear_r) / 2.0
            ear_vals.append(ear)
            tier2_verdicts.append("PASS" if ear > 0.15 else "FAIL")
        else:
            tier2_verdicts.append("N/A")

        # Tier 3: Forensic (Texture)
        if face.face_crop_raw is not None:
            tex_score, tex_susp, _ = compute_texture_score(face.face_crop_raw)
            texture_vals.append(tex_score)
            tier3_verdicts.append("PASS" if not tex_susp else "FAIL")
        else:
            tier3_verdicts.append("N/A")

        collected += 1
        attempts += 1

    cam.release()
    pipeline.release()

    result["frames_analyzed"] = collected

    if collected == 0:
        result["flags"].append("NO_DATA: Could not collect any frames with faces")
        print("  ❌ No frames with faces collected!")
        return result

    # Tier 1 Summary
    real_count = tier1_verdicts.count("REAL")
    fake_count = tier1_verdicts.count("FAKE")
    result["tier1_neural"] = {
        "real_count": real_count,
        "fake_count": fake_count,
        "always_real_pct": round(real_count / max(1, collected) * 100, 1),
        "confidence_mean": round(float(np.mean(neural_confs)), 4) if neural_confs else None,
        "confidence_std": round(float(np.std(neural_confs)), 4) if neural_confs else None,
    }
    print(f"\n  Tier 1 (Neural):  REAL={real_count} FAKE={fake_count} ({real_count / max(1, collected) * 100:.0f}% REAL)")
    if neural_confs:
        print(f"    Confidence: mean={np.mean(neural_confs):.4f} std={np.std(neural_confs):.4f}")

    if real_count == collected and fake_count == 0:
        result["flags"].append(
            "TIER1_ALWAYS_REAL: Neural network says REAL for every single frame. "
            "If you showed a deepfake, this is the bug."
        )
        print("  ❌ FLAG: Neural ALWAYS says REAL — potential logic collapse")

    # Tier 2 Summary
    pass_count = tier2_verdicts.count("PASS")
    fail_count = tier2_verdicts.count("FAIL")
    result["tier2_liveness"] = {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "ear_mean": round(float(np.mean(ear_vals)), 4) if ear_vals else None,
        "ear_std": round(float(np.std(ear_vals)), 4) if ear_vals else None,
    }
    print(f"  Tier 2 (Liveness): PASS={pass_count} FAIL={fail_count}")
    if ear_vals:
        print(f"    EAR: mean={np.mean(ear_vals):.4f} std={np.std(ear_vals):.4f}")

    if fail_count == collected:
        result["flags"].append(
            "TIER2_ALWAYS_FAIL: Liveness ALWAYS fails. "
            "EAR values may be too low or threshold too high."
        )
        print("  ❌ FLAG: Liveness ALWAYS fails — blink detection is broken")

    # Tier 3 Summary
    t3_pass = tier3_verdicts.count("PASS")
    t3_fail = tier3_verdicts.count("FAIL")
    result["tier3_forensic"] = {
        "pass_count": t3_pass,
        "fail_count": t3_fail,
        "texture_mean": round(float(np.mean(texture_vals)), 4) if texture_vals else None,
        "texture_std": round(float(np.std(texture_vals)), 4) if texture_vals else None,
    }
    print(f"  Tier 3 (Forensic): PASS={t3_pass} FAIL={t3_fail}")
    if texture_vals:
        print(f"    Texture score: mean={np.mean(texture_vals):.4f} std={np.std(texture_vals):.4f}")

    result["passed"] = len(result["flags"]) == 0

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def diagnose_logic_collapse(
    use_camera: bool = True,
    camera_id: int = 0,
    ear_duration: int = 10,
) -> dict[str, Any]:
    """Run all 5 diagnostics and produce a comprehensive report.

    Saves results to: diagnostics/logic_collapse_report.json

    Args:
        use_camera: If False, skip camera-dependent diagnostics.
        camera_id: Camera device index.
        ear_duration: Seconds for EAR stream capture.

    Returns:
        Full diagnostic report dict.
    """
    print("\n" + "═" * 60)
    print("  SHIELD-RYZEN V2 — LOGIC COLLAPSE DIAGNOSTIC")
    print("  Timestamp: " + datetime.now(timezone.utc).isoformat())
    print("═" * 60)

    report: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "camera_used": use_camera,
            "camera_id": camera_id,
        },
        "diagnostics": {},
        "summary": {
            "all_flags": [],
            "critical_count": 0,
            "verdict": "UNKNOWN",
        },
    }

    # Capture one live frame for diagnostics 1 & 2
    live_frame = None
    if use_camera:
        try:
            from shield_camera import ShieldCamera
            cam = ShieldCamera(camera_id=camera_id)
            for _ in range(10):  # Read a few frames to flush buffer
                ok, frame, _ = cam.read_validated_frame()
                if ok:
                    live_frame = frame
            cam.release()
        except Exception as e:
            print(f"  ⚠️  Could not capture live frame: {e}")

    # Run all diagnostics
    report["diagnostics"]["normalization"] = _diag_normalization(live_frame)
    report["diagnostics"]["landmarks"] = _diag_landmarks(live_frame)
    report["diagnostics"]["ear_stream"] = _diag_ear_stream(camera_id, ear_duration, use_camera)
    report["diagnostics"]["model_output"] = _diag_model_output(camera_id, 100, use_camera)
    report["diagnostics"]["tier_isolation"] = _diag_tier_isolation(camera_id, 30, use_camera)

    # Aggregate all flags
    all_flags = []
    for diag_name, diag_result in report["diagnostics"].items():
        flags = diag_result.get("flags", [])
        for f in flags:
            all_flags.append(f"[{diag_name.upper()}] {f}")

    report["summary"]["all_flags"] = all_flags
    report["summary"]["critical_count"] = len(all_flags)

    if len(all_flags) == 0:
        report["summary"]["verdict"] = "ALL_CLEAR"
    elif any("CRITICAL" in f or "STUCK" in f or "GARBAGE" in f for f in all_flags):
        report["summary"]["verdict"] = "CRITICAL_FAILURE"
    else:
        report["summary"]["verdict"] = "ISSUES_FOUND"

    # Save report
    report_path = os.path.join(_DIAG_DIR, "logic_collapse_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "═" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("═" * 60)
    print(f"  Verdict: {report['summary']['verdict']}")
    print(f"  Total flags: {len(all_flags)}")

    if all_flags:
        print("\n  All flags:")
        for flag in all_flags:
            print(f"    🚩 {flag}")
    else:
        print("  ✅ No issues detected")

    print(f"\n  📋 Full report saved → {report_path}")
    print("═" * 60)

    return report


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shield-Ryzen V2 — Logic Collapse Emergency Diagnostic"
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Skip camera-dependent diagnostics (use synthetic data only)"
    )
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--duration", type=int, default=10,
        help="Duration for EAR stream capture in seconds (default: 10)"
    )
    args = parser.parse_args()

    diagnose_logic_collapse(
        use_camera=not args.no_camera,
        camera_id=args.camera_id,
        ear_duration=args.duration,
    )
