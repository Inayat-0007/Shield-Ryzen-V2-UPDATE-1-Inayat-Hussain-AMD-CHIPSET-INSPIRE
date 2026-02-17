"""
Shield-Ryzen Level 3 — INT8 NPU Engine
======================================
Executes the quantized shield_ryzen_int8.onnx model.
Optimized for AMD Ryzen AI NPU (via QDQ format).
Currently running on NVIDIA RTX 3050 (simulating NPU execution).

Metrics:
- Model Size: ~20 MB (vs 79 MB FP32)
- Operations: INT8 (Quantized)
- Logic: Security Mode (89% Rule + Blink + Texture)

Developer: Inayat Hussain | AMD Slingshot 2026
"""

import cv2
import numpy as np
import torch  # CUDA DLL loading (Windows requirement for ORT CUDA provider)
import onnxruntime as ort
import mediapipe as mp
import os
import time

from shield_utils import (
    preprocess_face, calculate_ear, check_texture, classify_face,
    setup_logger, CONFIG,
    CONFIDENCE_THRESHOLD, BLINK_THRESHOLD, BLINK_TIME_WINDOW,
    LAPLACIAN_THRESHOLD, LEFT_EYE, RIGHT_EYE,
)

log = setup_logger('ShieldV3')

# ─── 1. Setup INT8 Session ───────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, 'shield_ryzen_int8.onnx')

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
    active_provider = session.get_providers()[0]
except Exception as e:
    log.critical("Error loading INT8 model: %s", e)
    exit(1)

model_size = os.path.getsize(onnx_path) / 1024 / 1024
log.info("💎 Shield-Ryzen Level 3 Active")
log.info("   Model:    shield_ryzen_int8.onnx (%.2f MB)", model_size)
log.info("   Provider: %s", active_provider)

# ─── 2. Setup MediaPipe ──────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_cfg = CONFIG['mediapipe']
landmarker_path = os.path.join(script_dir, mp_cfg['landmarker_model'])
landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=landmarker_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=mp_cfg['num_faces'],
    min_face_detection_confidence=mp_cfg['min_face_detection_confidence'],
    min_face_presence_confidence=mp_cfg['min_face_presence_confidence'],
    min_tracking_confidence=mp_cfg['min_tracking_confidence'],
)

# ─── 3. Start Security Mode ─────────────────────────────────
cap = cv2.VideoCapture(0)

log.info("═" * 55)
log.info("  💎 SHIELD-RYZEN LEVEL 3 — INT8 ENGINE ACTIVE")
log.info("═" * 55)
log.info("  Confidence: %d%%", CONFIDENCE_THRESHOLD * 100)
log.info("  Blink Win:  %ds", BLINK_TIME_WINDOW)
log.info("  Format:     INT8 (Quantized)")
log.info("  Press ESC to exit.")
log.info("═" * 55)

frame_timestamp_ms = 0
fps_counter = 0
fps_display = 0.0
fps_timer = time.time()
inference_ms = 0.0

blink_count = 0
blink_timestamps = []
was_eye_closed = False

with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # FPS
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

                # EAR
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
                if face_crop.size == 0: continue

                texture_score = check_texture(face_crop)
                texture_ok = texture_score > LAPLACIAN_THRESHOLD

                # INT8 INFERENCE
                input_tensor = preprocess_face(face_crop)

                inf_start = time.perf_counter()
                output = session.run(None, {'input': input_tensor})[0]
                inference_ms = (time.perf_counter() - inf_start) * 1000

                fake_prob = float(output[0, 0])
                real_prob = float(output[0, 1])

                # Security Classification
                label, color, tier = classify_face(fake_prob, real_prob, liveness_ok, texture_ok)

                # Draw
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"INT8 Real:{real_prob*100:.1f}%", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                info_x = x_max + 5
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (info_x, y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"INF: {inference_ms:.1f}ms", (info_x, y_min + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # HUD
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps_display:.1f} | INT8 ENGINE | {model_size:.1f} MB", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        blink_color = (0, 255, 0) if blink_count > 0 else (0, 0, 255)
        cv2.putText(frame, f"Blink: {blink_count} (10s window)", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)

        cv2.putText(frame, "DIAMOND TIER: NPU READY", (w - 220, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"Quantized: 20MB", (w - 150, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow('Shield-Ryzen Level 3 | INT8 Diamond Tier', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
log.info("Shield-Ryzen Level 3 — Session ended.")
