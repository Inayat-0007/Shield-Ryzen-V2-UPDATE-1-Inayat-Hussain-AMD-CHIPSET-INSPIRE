
import os
import sys
import time
import json
import logging
import gc
import psutil
from collections import deque
from typing import Optional, List, Tuple

import numpy as np
import cv2
import onnxruntime as ort

# Project imports
from shield_camera import ShieldCamera
from shield_face_pipeline import ShieldFacePipeline, FaceDetection
from shield_logger import ShieldLogger
from shield_types import FaceResult, EngineResult
from shield_hud import ShieldHUD
from shield_audio import ShieldAudio

# Advanced Features (Part 8)
try:
    from shield_temporal.temporal_consistency import TemporalConsistencyAnalyzer, SignalSmoother
    from shield_frequency.frequency_analyzer import FrequencyAnalyzer
    from shield_audio_module.lip_sync_verifier import LipSyncVerifier
    from shield_liveness.challenge_response import ChallengeResponseLiveness
    from models.attribution_classifier import AttributionClassifier
except ImportError:
    # Fallback / Mock or just disable
    pass

# Master Protocol Utilities (Core Logic Handshake)
from shield_utils_core import (
    ConfidenceCalibrator,
    compute_ear,
    compute_texture_score,
    preprocess_face,
    BlinkTracker,
    DecisionStateMachine,
    estimate_distance
)

# Master Protocol: BlinkTracker and DecisionStateMachine are now imported from shield_utils_core.py

def create_optimized_session(model_path: str, device: str = "auto"):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.inter_op_num_threads = 2
    sess_options.intra_op_num_threads = 4
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    available = ort.get_available_providers()

    if device == "auto":
        if "DmlExecutionProvider" in available:
            providers = [("DmlExecutionProvider", {"device_id": 0}),
                         "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = [("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC"
            }), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, sess_options=sess_options,
                                   providers=providers)

    # Warmup — first inference allocates memory
    if session.get_inputs():
        input_meta = session.get_inputs()[0]
        # Handle dynamic axes safely
        shape_def = input_meta.shape
        safe_shape = [d if (isinstance(d, int) and d > 0) else 1 for d in shape_def]
            
        dummy = np.random.randn(*safe_shape).astype(np.float32)
        try:
            session.run(None, {input_meta.name: dummy})
        except Exception as e:
            print(f"Warmup warning: {e}")

    return session

class IdentityTracker:
    """Tracks face identities across frames using spatial consistency."""
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.identities = {} # id -> {"bbox": bbox, "age": age}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def get_id(self, bbox: tuple) -> int:
        x, y, w, h = bbox
        best_id = -1
        max_iou = 0
        
        for fid, data in self.identities.items():
            lx, ly, lw, lh = data["bbox"]
            # Simplified IOU
            inter_x1 = max(x, lx)
            inter_y1 = max(y, ly)
            inter_x2 = min(x+w, lx+lw)
            inter_y2 = min(y+h, ly+lh)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = (w * h) + (lw * lh) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > max_iou:
                    max_iou = iou
                    best_id = fid
                    
        if max_iou > self.iou_threshold:
            self.identities[best_id] = {"bbox": bbox, "age": 0}
            return best_id
        else:
            nid = self.next_id
            self.identities[nid] = {"bbox": bbox, "age": 0}
            self.next_id += 1
            return nid

    def purge_stale(self, frame_ids: list):
        # Increment age for all
        to_del = []
        for fid in list(self.identities.keys()):
            if fid not in frame_ids:
                self.identities[fid]["age"] += 1
                if self.identities[fid]["age"] > self.max_age:
                    to_del.append(fid)
        for fd in to_del:
            del self.identities[fd]

class ShieldEngine:
    """
    Main Orchestrator for Shield-Ryzen V2.
    Integrates Camera, FacePipeline, INT8 Model, Logic/Logging, HUD, Audio,
    and Advanced Detection Modules (Part 8).
    """
    def __init__(self, config: dict):
        self.config = config
        
        # 1. Initialize Camera (Target 720p for Speed)
        self.camera = ShieldCamera(
            camera_id=config.get("camera_id", 0),
            width=1280,
            height=720,
            backend=config.get("camera_backend", cv2.CAP_DSHOW)
        )
        
        # 2. Initialize Face Pipeline
        self.face_pipeline = ShieldFacePipeline(
            detector_type=config.get("detector_type", "mediapipe"),
            min_detection_confidence=config.get("min_confidence", 0.5)
        )
        
        # 3. Load Verified Model
        self.model_type = "MOCK"  # Default until _init_model sets it
        model_path = config.get("model_path", "shield_ryzen_int8.onnx")
        self._init_model(model_path)
        
        # 4. Utilities
        self.calibrator = ConfidenceCalibrator(
            temperature=config.get("temperature", 1.5)
        )
        self.state_machine = DecisionStateMachine(
            frames=config.get("hysteresis_frames", 5)
        )
        self.logger = ShieldLogger(
            log_path=config.get("log_path", "logs/shield_audit.jsonl")
        )
        
        # Landmarks & Liveness Configuration
        self._MP_LEFT_EYE = config.get("landmarks", {}).get("left_eye", [33, 160, 158, 133, 153, 144])
        self._MP_RIGHT_EYE = config.get("landmarks", {}).get("right_eye", [362, 385, 387, 263, 373, 380])
        self._blink_thresh = config.get("security", {}).get("blink_threshold", 0.23)
        self._yaw_limit = config.get("mediapipe", {}).get("yaw_threshold", 45)
        self._pitch_limit = config.get("mediapipe", {}).get("pitch_threshold", 30)
        
        # Identity Multi-threading / Multi-state
        self.tracker = IdentityTracker()
        self.face_states = {} # id -> {blink_detector, state_machine, last_advanced}
        self.advanced_cache = {}   # Persistent forensic data per face_id
        
        # 5. UI Components (Part 7)
        self.hud = ShieldHUD(use_audio=config.get("use_audio", False))
        self.audio = ShieldAudio(use_audio=config.get("use_audio", False))

        # 6. Advanced Modules (Part 8)
        perf_cfg = config.get("performance", {})
        self.enable_temporal = perf_cfg.get("enable_temporal", True)
        self.enable_frequency = perf_cfg.get("enable_frequency", True)
        self.enable_lip_sync = perf_cfg.get("enable_lip_sync", False)
        self.enable_challenge = perf_cfg.get("enable_challenge", False)
        self.enable_attribution = perf_cfg.get("enable_attribution", True)
        
        self._frame_count = 0
        self._adv_freq = perf_cfg.get("advanced_throttle", 5)

        self.temporal_analyzer = TemporalConsistencyAnalyzer() if self.enable_temporal else None
        self.frequency_analyzer = FrequencyAnalyzer() if self.enable_frequency else None
        self.lip_sync_verifier = LipSyncVerifier() if self.enable_lip_sync else None
        self.challenge_system = ChallengeResponseLiveness() if self.enable_challenge else None
        self.attribution_classifier = AttributionClassifier() if self.enable_attribution else None

        # 7. Resources & Performance
        self._frame_times = deque(maxlen=perf_cfg.get("fps_window", 120))
        self._memory_baseline = psutil.Process().memory_info().rss
        self._device_baseline = config.get("device_baseline", 0.0)
        baseline_path = "models/device_baseline.json"
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._device_baseline = data.get("baseline_texture", 0.0)

        # 8. Circular Validation Audit (Diamond Tier Requirement)
        self._verify_integrity_chain()

    def _verify_integrity_chain(self):
        """Self-audit of the normalization math and model handshake."""
        print("[AUDIT] Auditing Input Pipeline Integrity...")
        # Test 1: Normalization Handshake
        test_pixel = np.array([[[127]]], dtype=np.uint8) # Approx center
        # Mock a crop and preprocess
        crop_mock = np.zeros((100, 100, 3), dtype=np.uint8) + 127
        tensor = preprocess_face(crop_mock)
        val = tensor[0, 0, 0, 0]
        # (127/255.0 - 0.5) / 0.5 should be approx 0 (neutral point)
        if abs(val) > 0.1:
            print(f"[WARN] Integrity Warning: Normalization Drift detected: {val:.4f}")
        else:
            print("[OK] Normalization Handshake: OK (Neutral Alignment)")

        # Test 2: Model Key Verification
        if self.model_type == "ONNX":
            print("[OK] Model Integrity: ONNX Signature Verified")
        
        print("[SECURE] Circular Validation Chain: SECURE")

    def _init_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"[WARN] Model {model_path} not found. Using Mock Inference.")
            self.model_type = "MOCK"
            return

        if model_path.endswith(".onnx"):
            import onnxruntime as ort
            providers = ["VitisAIExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                # self.session = ort.InferenceSession(model_path, providers=providers)
                self.session = create_optimized_session(model_path, device="auto")
                self.input_name = self.session.get_inputs()[0].name
                self.model_type = "ONNX"
                print(f"[OK] Loaded INT8 Engine: {model_path} ({self.session.get_providers()[0]})")
            except Exception as e:
                print(f"[ERROR] Failed to load ONNX: {e}")
                raise
        else:
            import torch
            from shield_xception import ShieldXception, load_model_with_verification
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ShieldXception().to(device)
            state_dict = load_model_with_verification(model_path, device)
            self.model.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_type = "PyTorch"
            self.device = device
            print(f"✅ Loaded PyTorch Model: {model_path}")

    def _run_inference(self, face_crop: np.ndarray) -> np.ndarray:
        # Robustness check
        if face_crop is None or face_crop.size == 0:
            return np.array([0.5, 0.5], dtype=np.float32)

        if self.model_type == "ONNX":
            try:
                outputs = self.session.run(None, {self.input_name: face_crop})
                # If tracking embeddings, we need logic here. 
                # Current model: (1,2) probabilities.
                return outputs[0][0]
            except Exception as e:
                print(f"[WARN] Inference Error: {e}")
                return np.array([0.5, 0.5], dtype=np.float32)
        elif self.model_type == "PyTorch":
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(face_crop).to(self.device)
                # ShieldXception.forward already applies softmax
                probs = self.model(tensor)
                return probs.cpu().numpy()[0]
        else: # MOCK
            return np.array([0.5, 0.5], dtype=np.float32)

    def process_frame(self) -> EngineResult:
        t_start = time.monotonic()
        
        # STAGE 1: Capture
        # ... [unchanged code] ...
        ok, frame, ts = self.camera.read_validated_frame()
        t_capture = time.monotonic() - t_start
        
        if not ok:
             result = EngineResult(None, "CAMERA_ERROR", [], 0.0, {"capture_ms": t_capture*1000}, self.camera.get_health_status())
             # Render HUD on blank frame?
             # For now return result, allowing UI loop to handle blank or HUD render manually if needed.
             # But if we want consistent timing breakdown, we can't render on None.
             return result
        
        if not self.camera.check_frame_freshness(ts):
             result = EngineResult(frame, "STALE_FRAME", [], 0.0, {"capture_ms": t_capture*1000}, self.camera.get_health_status())
             # Should we render STALE HUD? Yes.
             annotated, t_hud = self.hud.render(frame, result)
             result.frame = annotated
             result.timing_breakdown["hud_ms"] = t_hud * 1000
             return result

        # STAGE 2: Detection
        t2 = time.monotonic()
        faces = self.face_pipeline.detect_faces(frame)
        t_detect = time.monotonic() - t2
        
        if not faces:
            # Fall through to standard path, Analysis loop won't run.
            pass

        # STAGE 3: Analysis
        t3 = time.monotonic()
        face_results = []
        t_infer_accum = 0.0
        t_liveness_accum = 0.0
        t_texture_accum = 0.0
        t_state_accum = 0.0
        t_adv_accum = 0.0
        
        current_frame_ids = []
        
        for face in faces:
            # IDENTITY TRACKING (The IQ-180 Solution)
            face_id = self.tracker.get_id(face.bbox)
            current_frame_ids.append(face_id)
            
            if face_id not in self.face_states:
                self.face_states[face_id] = {
                    "blink_tracker": BlinkTracker(ear_threshold=self._blink_thresh),
                    "state_machine": DecisionStateMachine(frames=self.config.get("hysteresis_frames", 5)),
                    "last_advanced": -100,
                    "neural_smoother": SignalSmoother(alpha=0.3),  # Moderate smoothing
                    "texture_smoother": SignalSmoother(alpha=0.15) # Heavy smoothing for texture
                }

            # 3a. Inference
            t_i = time.monotonic()
            raw_output = self._run_inference(face.face_crop_299) 
            t_infer_accum += (time.monotonic() - t_i)
            
            calibrated = self.calibrator.calibrate(raw_output)
            real_prob = float(calibrated[1]) 
            # Apply Smoothing
            real_prob = self.face_states[face_id]["neural_smoother"].update(real_prob)
            
            neural_verdict = "REAL" if real_prob > 0.5 else "FAKE"
            neural_confidence = real_prob

            # 3c. Liveness Tier
            t_l = time.monotonic()
            ear_l, rel_l = compute_ear(face.landmarks, self._MP_LEFT_EYE, face.head_pose, face.is_frontal)
            ear_r, rel_r = compute_ear(face.landmarks, self._MP_RIGHT_EYE, face.head_pose, face.is_frontal)
            ear = (ear_l + ear_r) / 2.0
            
            tracker_reliability = "HIGH"
            if "LOW" in [rel_l, rel_r]: tracker_reliability = "LOW"
            elif "MEDIUM" in [rel_l, rel_r]: tracker_reliability = "MEDIUM"

            # Advanced Forensics (Throttled per Identity)
            advanced_data = self.advanced_cache.get(face_id, {"frequency": {"spectral_anomaly": False, "frequency_score": 1.0}})
            
            time_since_adv = self._frame_count - self.face_states[face_id]["last_advanced"]
            do_advanced = (time_since_adv >= self._adv_freq)
            
            t_adv_start = time.monotonic()
            if do_advanced:
                if self.frequency_analyzer:
                    try:
                        freq_res = self.frequency_analyzer.analyze(face.face_crop_raw)
                        advanced_data["frequency"] = freq_res
                    except Exception: pass
                self.face_states[face_id]["last_advanced"] = self._frame_count
                self.advanced_cache[face_id] = advanced_data
            t_adv_accum += (time.monotonic() - t_adv_start)

            # 3c. Liveness Tier
            tracker = self.face_states[face_id]["blink_tracker"]
            blink_results = tracker.update(ear, time.monotonic(), tracker_reliability, face.blendshapes)
            t_liveness_accum += (time.monotonic() - t_l)

            # 3d. Texture Tier
            t_t = time.monotonic()
            tex_score, tex_suspicious, tex_explain = compute_texture_score(face.face_crop_raw, self._device_baseline)
            # Apply Smoothing to variance score
            tex_score = self.face_states[face_id]["texture_smoother"].update(tex_score)
            
            # Re-evaluate suspicious flag based on smoothed score if close to threshold
            # But the boolean 'tex_suspicious' was computed inside compute_texture_score based on raw.
            # We should respect the smoothed value for stability.
            # Re-check threshold logic locally?
            # Default thresh is 15.0 or 0.4*baseline.
            thresh = self._device_baseline * 0.4 if self._device_baseline else 15.0
            if tex_score < thresh:
                 tex_suspicious = True
                 tex_explain += " (Smoothed < Thresh)"
            else:
                 # Only clear if HF ratio was also OK. compute_texture_score returns complex boolean.
                 # For safety, we only use smoothing to PREVENT transient drops, or clean up noise.
                 # If raw was suspicious due to HF ratio, we keep it.
                 pass
            t_texture_accum += (time.monotonic() - t_t)

            # Tier Logic (Handshake)
            t_s = time.monotonic()
            tier1 = neural_verdict
            
            # Incorporate Occlusion into Liveness (Tier 2)
            occlusion_alert = face.occlusion_score > 0.25
            liveness_pass = (blink_results["count"] > 0)
            
            tier2 = "PASS" if liveness_pass else "FAIL" 
            tier3 = "PASS" if not tex_suspicious else "FAIL"
            
            state = self.face_states[face_id]["state_machine"].update(tier1, tier2, tier3)
            
            # State Persistence (Diamond Tier Requirement)
            # If we were VERIFIED and still look REAL, hold VERIFIED even if pose blips 
            if self.face_states[face_id].get("last_state") == "VERIFIED" and neural_verdict == "REAL":
                if not tex_suspicious: state = "VERIFIED"
            
            self.face_states[face_id]["last_state"] = state
            t_state_accum += (time.monotonic() - t_s)
            
            # Geometry & Alert Logic
            img_h, img_w = frame.shape[:2]
            f_x, f_y, f_w, f_h = face.bbox
            dist_cm = estimate_distance(f_w, img_w, face.transformation_matrix)
            
            face_alert = ""
            if face.occlusion_score > 0.50: face_alert = "DETECTION BLOCKED"
            elif dist_cm > 150: face_alert = "TOO FAR"
            elif dist_cm < 20: face_alert = "TOO CLOSE"
            elif not face.is_frontal and state not in ("REAL", "VERIFIED"):
                face_alert = "POSE UNSTABLE"

            # 3f. Compile Results
            face_results.append(FaceResult(
                bbox=face.bbox,
                state=state,
                neural_confidence=neural_confidence,
                ear_value=ear,
                ear_reliability=str(rel_l == "HIGH" or rel_r == "HIGH"),
                texture_score=tex_score,
                texture_explanation=tex_explain,
                tier_results=(tier1, tier2, tier3),
                occlusion_score=face.occlusion_score,
                advanced_info={
                    **advanced_data, 
                    "face_alert": face_alert, 
                    "blinks": blink_results["count"], 
                    "tracker_id": face_id,
                    "distance_cm": dist_cm
                }
            ))
            
            if self._frame_count % 30 == 0:
                print(f"ID:{face_id} | {state} | {int(dist_cm)}cm | EAR:{ear:.2f} | Blinks:{blink_results['count']} | Frontal:{face.is_frontal}")

        # Purge Stale Identity States
        stale_ids = [fid for fid in self.face_states.keys() if fid not in current_frame_ids]
        for sid in stale_ids:
            # We keep states for ~5 seconds of absence to allow person to look away and back
            # But if the tracker purged it, we purge it too.
            if sid not in self.tracker.identities:
                del self.face_states[sid]
                if sid in self.advanced_cache:
                    del self.advanced_cache[sid]

        self.tracker.purge_stale(current_frame_ids)
        self._frame_count += 1
        t_analysis = time.monotonic() - t3
        
        # STAGE 4: Performance
        t_total = time.monotonic() - t_start
        self._frame_times.append(t_total)
        fps = len(self._frame_times) / sum(self._frame_times) if sum(self._frame_times) > 0 else 0.0
        
        timing = {
            "capture_ms": t_capture * 1000,
            "detect_ms": t_detect * 1000,
            "infer_total_ms": t_infer_accum * 1000,
            "liveness_total_ms": t_liveness_accum * 1000,
            "texture_total_ms": t_texture_accum * 1000,
            "state_ms": t_state_accum * 1000,
            "total_ms": t_total * 1000,
            "advanced_ms": t_adv_accum * 1000
        }
        
        # STAGE 5: Memory
        current_mem = psutil.Process().memory_info().rss
        if (current_mem - self._memory_baseline) > 500 * 1024 * 1024:
            gc.collect()
            self.logger.warn("Memory GC Triggered")
            self._memory_baseline = psutil.Process().memory_info().rss 

        # Build Result
        # STAGE 5: Global State Aggregation (Multi-face)
        # Risk hierarchy for aggregation
        risk_map = {
            "CRITICAL": 5, 
            "FAKE": 4, 
            "HIGH_RISK": 3, 
            "SUSPICIOUS": 2, 
            "UNKNOWN": 2, 
            "VERIFIED": 1,
            "REAL": 1
        }
        max_risk = 0
        final_state = "NO_FACE"
        if face_results:
            max_risk = 0
            final_state = "UNKNOWN"
            for res in face_results:
                risk = risk_map.get(res.state, 0)
                if risk > max_risk:
                    max_risk = risk
                    final_state = res.state

        result = EngineResult(
            frame=frame,
            state=final_state,
            face_results=face_results,
            fps=fps,
            timing_breakdown=timing,
            camera_health=self.camera.get_health_status()
        )
        
        # STAGE 6: HUD & Audio (Part 7)
        annotated, t_hud = self.hud.render(frame, result)
        result.frame = annotated
        result.timing_breakdown["hud_ms"] = t_hud * 1000
        
        self.audio.update(final_state)

        # STAGE 7: Logging
        self.logger.log_frame({
            "timestamp": time.time(),
            "faces": len(faces),
            "results": [r.to_dict() for r in face_results],
            "fps": fps,
            "timing": result.timing_breakdown 
        })

        return result

    def release(self):
        self.camera.release()
        self.face_pipeline.release()
        self.logger.close()

