"""
Shield-Ryzen V2 — Universal ONNX Engine (SECURITY MODE)
========================================================
Zero PyTorch dependency. Pure ONNX Runtime + NumPy + OpenCV.
Optimized for NVIDIA RTX 3050 (CUDAExecutionProvider)
and AMD Ryzen AI NPU ready.

Developer: Inayat Hussain | AMD Slingshot 2026
"""

import cv2
import numpy as np
import torch  # Load CUDA DLLs into process (required for ORT CUDA provider on Windows)
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

log = setup_logger('ShieldV2')

# ─── 1. Setup ONNX Runtime Session (GPU) ─────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, 'shield_ryzen_v2.onnx')

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
active_provider = session.get_providers()[0]

log.info("Provider: %s", active_provider)
log.info("Model:    %s", onnx_path)
log.info("Input:    %s", session.get_inputs()[0].shape)

# ─── 2. Setup MediaPipe FaceLandmarker ────────────────────────
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

# ─── 3. Start Security Mode ──────────────────────────────────
cap = cv2.VideoCapture(0)

log.info("═" * 55)
log.info("  🛡️  SHIELD-RYZEN V2 — ONNX SECURITY MODE")
log.info("═" * 55)
log.info("  Engine:     ONNX Runtime %s (NO PyTorch)", ort.__version__)
log.info("  Provider:   %s", active_provider)
log.info("  Threshold:  %d%%", CONFIDENCE_THRESHOLD * 100)
log.info("  Blink:      %ds window", BLINK_TIME_WINDOW)
log.info("  Texture:    Laplacian > %d", LAPLACIAN_THRESHOLD)
log.info("  Press ESC to exit.")
log.info("═" * 55)

frame_timestamp_ms = 0
fps_counter = 0
fps_display = 0.0
fps_timer = time.time()
inference_ms = 0.0

# Blink tracking
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

        # Detect landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        lm_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        h, w, _ = frame.shape

        # Prune old blinks
        blink_timestamps = [t for t in blink_timestamps if now - t < BLINK_TIME_WINDOW]
        blink_count = len(blink_timestamps)

        if lm_result.face_landmarks:
            for face_landmarks in lm_result.face_landmarks:
                # Bounding box from landmarks
                xs = [lm.x for lm in face_landmarks]
                ys = [lm.y for lm in face_landmarks]
                x_min = max(0, int(min(xs) * w) - 10)
                y_min = max(0, int(min(ys) * h) - 10)
                x_max = min(w, int(max(xs) * w) + 10)
                y_max = min(h, int(max(ys) * h) + 10)

                # EAR Blink Detection
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

                # Face crop
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    continue

                # Texture guard
                texture_score = check_texture(face_crop)
                texture_ok = texture_score > LAPLACIAN_THRESHOLD

                # ── ONNX Inference (GPU) ──
                input_tensor = preprocess_face(face_crop)

                inf_start = time.perf_counter()
                output = session.run(None, {'input': input_tensor})[0]
                inference_ms = (time.perf_counter() - inf_start) * 1000

                # Class mapping: Index 0 = Fake, Index 1 = Real
                fake_prob = float(output[0, 0])
                real_prob = float(output[0, 1])

                # ══ SECURITY MODE CLASSIFICATION ══
                label, color, tier = classify_face(fake_prob, real_prob, liveness_ok, texture_ok)

                # Draw
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Real:{real_prob*100:.1f}% Fake:{fake_prob*100:.1f}%",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Per-face stats
                info_x = x_max + 5
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (info_x, y_min + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"TEX: {texture_score:.0f}", (info_x, y_min + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"INF: {inference_ms:.1f}ms", (info_x, y_min + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── HUD ──
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps_display:.1f} | ONNX | {active_provider}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        blink_color = (0, 255, 0) if blink_count > 0 else (0, 0, 255)
        blink_text = f"Blink: {'YES' if blink_count > 0 else 'NO'} ({blink_count} in {BLINK_TIME_WINDOW}s) | INF: {inference_ms:.1f}ms"
        cv2.putText(frame, blink_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)

        cv2.putText(frame, "V2 ONNX ENGINE", (w - 180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
        cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", (w - 170, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow('Shield-Ryzen V2 | ONNX SECURITY MODE', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
log.info("Shield-Ryzen V2 — Session ended.")
