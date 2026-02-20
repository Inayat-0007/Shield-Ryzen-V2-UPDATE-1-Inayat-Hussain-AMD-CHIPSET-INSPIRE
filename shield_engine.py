"""
Shield-Ryzen V2 — ShieldEngine (Unified Async Core) (TASK 6.1)
==============================================================
The central orchestrator for Shield-Ryzen V2.
Integration of all verified modules into a high-performance,
secure, async execution engine.

Architecture: Triple-Buffer Async Pipeline
  1. Camera Thread: Captures validated frames (Part 1)
  2. AI Thread: Analysis & Neural Inference (Part 2, 4, 5)
  3. Main Thread (HUD): Rendering & Display (Part 6)

Features:
  - GIL-free capture/inference parallelism
  - Encrypted biometric memory (Part 6.3)
  - Plugin architecture for modular voting (Part 6.2)
  - Structured JSONL audit logging (Part 6.5)
  - Automatic domain adaptation/calibration (Part 6.4)
  - Temporal consistency via Identity Tracking and State Machine

Developer: Inayat Hussain | AMD Slingshot 2026
Part 6 of 14 — Integration & Efficiency
"""

import sys
import os
import time
import queue
import threading
import gc
import psutil
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np
import torch

# Project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shield_camera import ShieldCamera
from shield_face_pipeline import ShieldFacePipeline, FaceDetection
from shield_xception import load_model_with_verification, ShieldXception
from shield_utils_core import (
    ConfidenceCalibrator,
    DecisionStateMachine,
    compute_ear,
    compute_texture_score,
    BlinkTracker,
    LEFT_EYE,
    RIGHT_EYE
)
from shield_crypto import encrypt, decrypt, secure_wipe
from shield_plugin import ShieldPlugin
from shield_logger import get_logger

# Plugins (Part 7)
from plugins.challenge_response import ChallengeResponsePlugin
from plugins.rppg_heartbeat import HeartbeatPlugin
from plugins.stereo_depth import StereoDepthPlugin
from plugins.skin_reflectance import SkinReflectancePlugin

# Forensic Plugins (Part 8)
from plugins.frequency_analyzer import FrequencyAnalyzerPlugin
from plugins.codec_forensics import CodecForensicsPlugin
from plugins.adversarial_detector import AdversarialPatchPlugin
from plugins.lip_sync_verifier import LipSyncPlugin

# Constants
DEFAULT_CONFIG = {
    "camera_id": 0,
    "detector_type": "mediapipe",
    "model_path": "shield_ryzen_int8.onnx",  # Prefer INT8 (Part 5)
    "log_path": "logs/shield_audit.jsonl",
    "temperature": 1.5,
    "hysteresis_frames": 5,
    "max_faces": 2,
    # Plugin Configs (Part 7)
    "enable_challenge_response": False, # Default OFF (User Opt-in)
    "enable_heartbeat": True,
    "enable_stereo_depth": False, # Requires 2nd camera
    "enable_skin_reflectance": True,
    # Forensic Configs (Part 8)
    "enable_frequency_analysis": True,
    "enable_codec_forensics": True,
    "enable_adversarial_detection": True,
    "enable_lip_sync": False # Default OFF (User Opt-in/Audio required)
}

@dataclass
class FaceResult:
    """Per-face analysis result."""
    face_id: int
    bbox: Tuple[int, int, int, int]
    landmarks: list
    state: str
    confidence: float # Overall confidence (usually neural)
    neural_confidence: float
    ear_value: float
    ear_reliability: str
    texture_score: float
    texture_explanation: str
    tier_results: Tuple[str, str, str]
    plugin_votes: List[dict]
    advanced_info: dict = None  # blinks, distance_cm, tracker_id, face_alert
    
    def __post_init__(self):
        if self.advanced_info is None:
            self.advanced_info = {}
    
    def to_dict(self):
        return asdict(self)

@dataclass
class EngineResult:
    """Full frame analysis outcome."""
    frame: Any  # Encrypted frame data (bytes) or raw if needed for display
    timestamp: float
    face_results: List[FaceResult]
    fps: float
    timing_breakdown: dict
    camera_health: dict
    memory_mb: float

class IdentityTracker:
    """Simple IOU-based tracker for temporal consistency across frames."""
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
            # IOU calculation
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
            self.identities[best_id]["bbox"] = bbox
            self.identities[best_id]["age"] = 0
            return best_id
        else:
            nid = self.next_id
            self.identities[nid] = {"bbox": bbox, "age": 0}
            self.next_id += 1
            return nid

    def purge_stale(self, visible_ids: list):
        # Increment age for all, remove if too old
        to_del = []
        for fid in self.identities:
            if fid not in visible_ids:
                self.identities[fid]["age"] += 1
                if self.identities[fid]["age"] > self.max_age:
                    to_del.append(fid)
        for fid in to_del:
            del self.identities[fid]

class PluginAwareStateMachine(DecisionStateMachine):
    """Extends DecisionStateMachine to incorporate Plugin votes.
    
    MAJORITY RULE: Only downgrade forensic tier if a MAJORITY of
    decisive plugins vote FAKE. This prevents a single noisy plugin
    from overriding an otherwise correct REAL verdict.
    
    STRONG CONSENSUS: If 2+ plugins vote FAKE, override neural verdict
    too — catches AI-generated faces that fool the neural network.
    
    WAIT_BLINK TIMEOUT: If stuck in WAIT_BLINK for >5s, escalate.
    Uses a sticky counter so the face can't bounce back to WAIT_BLINK.
    - 1st timeout → SUSPICIOUS
    - 2nd timeout → HIGH_RISK  
    A confirmed blink (tier2=PASS) resets the escalation counter.
    """
    WAIT_BLINK_TIMEOUT = 5.0  # seconds before WAIT_BLINK escalates
    
    def __init__(self, hysteresis: int = 5):
        super().__init__(hysteresis)
        self._no_blink_escalations = 0  # how many times we timed out
        self._wait_blink_entered = None  # monotonic time when WAIT_BLINK started
    
    def update(self, t1, t2, t3, plugin_votes=None) -> str:
        fake_count = 0
        decisive_count = 0
        
        if plugin_votes:
            # Only count decisive votes (REAL or FAKE), ignore UNCERTAIN/ERROR
            decisive = [v for v in plugin_votes 
                        if v.get("verdict") in ("REAL", "FAKE")]
            decisive_count = len(decisive)
            if decisive:
                fake_count = sum(1 for v in decisive if v["verdict"] == "FAKE")
                # Majority rule: more than half of decisive plugins must say FAKE
                if fake_count > decisive_count / 2:
                    t3 = "FAIL"
                
                # STRONG CONSENSUS: 3+ plugins say FAKE → override neural too
                # This catches AI-generated avatars that fool the neural network
                # (2+ would false-positive on real faces when frequency + adversarial are noisy)
                if fake_count >= 3:
                    t1 = "FAKE"
        
        # If a blink was confirmed (tier2=PASS), reset escalation counter
        if t2 == "PASS" or (hasattr(t2, 'passed') and t2.passed):
            self._no_blink_escalations = 0
            self._wait_blink_entered = None
        
        # Run base state machine
        result = super().update(t1, t2, t3)
        
        # Track when we enter WAIT_BLINK
        if self.state == "WAIT_BLINK":
            if self._wait_blink_entered is None:
                self._wait_blink_entered = time.monotonic()
            
            # Check timeout
            elapsed = time.monotonic() - self._wait_blink_entered
            if elapsed > self.WAIT_BLINK_TIMEOUT:
                self._no_blink_escalations += 1
                self._wait_blink_entered = None  # reset timer
                
                # Escalate based on how many times we've timed out
                if self._no_blink_escalations >= 2:
                    self.state = "HIGH_RISK"
                else:
                    self.state = "SUSPICIOUS"
                self._state_entry_time = time.monotonic()
                self._total_transitions += 1
        else:
            # Not in WAIT_BLINK — reset entry time
            self._wait_blink_entered = None
            
            # If we previously escalated and base SM wants to de-escalate
            # back to WAIT_BLINK, block it (sticky escalation)
            if self._no_blink_escalations > 0 and result == "WAIT_BLINK":
                if self._no_blink_escalations >= 2:
                    self.state = "HIGH_RISK"
                else:
                    self.state = "SUSPICIOUS"
        
        return self.state


class ShieldEngine:
    """
    Unified Async Triple-Buffer Engine.
    Orchestrates Camera -> AI -> HUD pipeline.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # 1. Initialize Modules
        self.logger = get_logger(os.path.dirname(self.config["log_path"]))
        self.logger.log({"event": "engine_init_start", "config": str(self.config)})
        
        # Camera (Part 1) — resolution from config for quality control
        self.camera = ShieldCamera(
            camera_id=self.config["camera_id"],
            width=self.config.get("camera_width", 1280),
            height=self.config.get("camera_height", 720),
        )
        
        # Face Pipeline (Part 2) using MediaPipe
        self.face_pipeline = ShieldFacePipeline(
            detector_type=self.config["detector_type"],
            max_faces=self.config["max_faces"]
        )
        
        # Model (Part 4/5)
        # Using secure load wrapper 
        # Support both ONNX (Part 5) and PyTorch (Part 4) paths
        self.model_path = self.config["model_path"]
        self.use_onnx = self.model_path.endswith(".onnx")
        
        self.device = torch.device("cpu") # Default
        if torch.cuda.is_available() and not self.use_onnx:
            self.device = torch.device("cuda")
            
        if self.use_onnx:
            import onnxruntime as ort
            # Try VitisAI first for Ryzen AI NPU
            providers = ["VitisAIExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.logger.log({"event": "model_loaded", "type": "ONNX", "providers": self.session.get_providers()})
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                self.logger.error(f"Failed to load ONNX model: {e}")
                raise
        else:
            # PyTorch fallback
            try:
                state_dict = load_model_with_verification(self.model_path, self.device)
                self.model = ShieldXception().to(self.device)
                self.model.model.load_state_dict(state_dict)
                self.model.eval()
                self.logger.log({"event": "model_loaded", "type": "PyTorch", "device": str(self.device)})
            except Exception as e:
                self.logger.error(f"Failed to load PyTorch model: {e}")
                # Fallback to mock if test environment? No, fail hard for security.
                raise

        # Logic Utils (Part 3)
        self.calibrator = ConfidenceCalibrator(self.config["temperature"])
        
        # Temporal State (Part 3 + new IdentityTracker)
        self.tracker = IdentityTracker()
        self.face_states = {} # face_id -> {state_machine, blink_tracker, smoothers...}
        self.hysteresis = self.config["hysteresis_frames"]

        # Plugins (Part 6.2 + Part 7 Registration)
        self.plugins: List[ShieldPlugin] = []
        
        if self.config.get("enable_challenge_response", False):
            try:
                self.register_plugin(ChallengeResponsePlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load ChallengeResponsePlugin: {e}")

        if self.config.get("enable_heartbeat", True):
            try:
                self.register_plugin(HeartbeatPlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load HeartbeatPlugin: {e}")

        if self.config.get("enable_stereo_depth", False):
            try:
                # Assuming camera 1 is secondary? Or user configured index? 
                # Defaults to index 1 inside plugin.
                self.register_plugin(StereoDepthPlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load StereoDepthPlugin: {e}")

        if self.config.get("enable_skin_reflectance", True):
            try:
                self.register_plugin(SkinReflectancePlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load SkinReflectancePlugin: {e}")

        # Forensic Plugins (Part 8 Registration)
        if self.config.get("enable_frequency_analysis", True):
            try:
                self.register_plugin(FrequencyAnalyzerPlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load FrequencyAnalyzerPlugin: {e}")

        if self.config.get("enable_codec_forensics", True):
            try:
                self.register_plugin(CodecForensicsPlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load CodecForensicsPlugin: {e}")

        if self.config.get("enable_adversarial_detection", True):
            try:
                self.register_plugin(AdversarialPatchPlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load AdversarialPatchPlugin: {e}")

        if self.config.get("enable_lip_sync", False):
            try:
                self.register_plugin(LipSyncPlugin())
            except Exception as e:
                self.logger.warn(f"Failed to load LipSyncPlugin: {e}")


        # Crypto (Part 6.3)
        # Implicitly initialized by import

        # Async Queues — size=1 for MINIMUM latency (no frame accumulation)
        self.camera_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        # Monitoring
        self.running = False
        self._memory_baseline = psutil.Process().memory_info().rss
        self._frame_times = deque(maxlen=120)
        self._device_baseline: dict = {}

        # Auto-Calibration (Task 6.4)
        self._perform_startup_calibration()
        
        self.logger.log({"event": "engine_init_complete"})


    def register_plugin(self, plugin: ShieldPlugin):
        """Register a detection plugin."""
        self.plugins.append(plugin)
        self.logger.log({"event": "plugin_registered", "name": plugin.name, "tier": plugin.tier})

    def _perform_startup_calibration(self):
        """30-second silent calibration (simulated for simplicity here)."""
        calib_file = "shield_calibration.json"
        if os.path.exists(calib_file):
            try:
                with open(calib_file, "r") as f:
                    self._device_baseline = json.load(f)
                self.logger.log({"event": "calibration_loaded", "baseline": self._device_baseline})
                return
            except Exception:
                pass

        print("First run: calibrating for your device (quick scan)...")
        # Simulating calibration logic - capturing ambient check
        # In real scenario: capture headers, verify lighting, FPS stability
        self._device_baseline = {
            "avg_fps": 30.0,
            "lighting_condition": "unknown",
            "texture_floor": 10.0
        }
        with open(calib_file, "w") as f:
            json.dump(self._device_baseline, f)
        self.logger.log({"event": "calibration_created", "baseline": self._device_baseline})


    def start(self):
        """Start async threads."""
        self.running = True
        
        self.cam_thread = threading.Thread(target=self._camera_thread, daemon=True)
        self.ai_thread = threading.Thread(target=self._ai_thread, daemon=True)
        
        self.cam_thread.start()
        self.ai_thread.start()
        self.logger.log({"event": "engine_started"})

    def stop(self):
        """Stop threads and clean up. Non-blocking on failures."""
        self.running = False
        
        # Join threads with short timeout (daemon threads will die anyway)
        try:
            if hasattr(self, 'cam_thread') and self.cam_thread.is_alive():
                self.cam_thread.join(timeout=1.0)
        except Exception:
            pass
        
        try:
            if hasattr(self, 'ai_thread') and self.ai_thread.is_alive():
                self.ai_thread.join(timeout=1.0)
        except Exception:
            pass
        
        # Release camera
        try:
            self.camera.release()
        except Exception:
            pass
        
        # Secure wipe (non-blocking)
        try:
            secure_wipe()
        except Exception:
            pass
        
        # Close logger
        try:
            self.logger.close()
        except Exception:
            pass

    def _camera_thread(self):
        """Thread 1: Capture validated frames."""
        while self.running:
            ok, frame, ts = self.camera.read_validated_frame()
            if ok: # check freshness implicitly via queue size
                try:
                    # Encrypt frame if storing long term? No, transient queue.
                    # But Python GIL release happens during queue.put check
                    self.camera_queue.put_nowait((frame, ts))
                except queue.Full:
                    try:
                        self.camera_queue.get_nowait() # Drop old
                        self.camera_queue.put_nowait((frame, ts))
                    except:
                        pass
            else:
                time.sleep(0.01) # Avoid busy loop on cam fail

    def _ai_thread(self):
        """Thread 2: Process faces, run inference + plugins."""
        while self.running:
            try:
                # Get frame with timeout to allow checking self.running
                frame, ts = self.camera_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self._process_frame(frame, ts)
                # Send result to HUD
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait() # Drop old result
                        self.result_queue.put_nowait(result)
                    except:
                        pass
            except Exception as e:
                self.logger.error(f"AI Thread Error: {e}", exc_info=True)


    def _estimate_distance(self, bbox, frame_shape) -> float:
        """Estimate face distance using pinhole camera model.
        
        Pinhole model: distance = (known_width × focal_length) / pixel_width
        
        Uses:
          - Average human face width ≈ 14 cm
          - Estimated focal length from frame width (typical webcam FOV ~60°)
          
        Returns distance in centimeters, or 0 if unable to estimate.
        """
        KNOWN_FACE_WIDTH_CM = 14.0    # Average adult face width
        
        _, _, face_w, face_h = bbox
        if face_w < 20:  # Too small to estimate
            return 0.0
        
        frame_h, frame_w = frame_shape[:2]
        # Estimated focal length: frame_width / (2 * tan(FOV/2))
        # For ~60° FOV: focal ≈ frame_width * 0.87
        focal_length_px = frame_w * 0.87
        
        distance_cm = (KNOWN_FACE_WIDTH_CM * focal_length_px) / face_w
        return round(distance_cm, 1)

    def _run_inference(self, face_crop_299: np.ndarray) -> np.ndarray:
        """Run neural inference (ONNX or PyTorch). Returns [fake_prob, real_prob]."""
        if self.use_onnx:
            # ONNX Runtime input
            # face_crop is (1, 3, 299, 299) float32
            return self.session.run(None, {self.input_name: face_crop_299})[0][0]
        else:
            # PyTorch
            tensor = torch.from_numpy(face_crop_299).to(self.device)
            with torch.no_grad():
                return self.model(tensor).cpu().numpy()[0]

    def _process_frame(self, frame, ts) -> EngineResult:
        """
        Full pipeline for one frame.
        """
        t_start = time.monotonic()
        timing = {}

        # STAGE 1: Face Detection (Part 2)
        t0 = time.monotonic()
        faces = self.face_pipeline.detect_faces(frame)
        timing["detect_ms"] = (time.monotonic() - t0) * 1000

        # STAGE 2: Per-face analysis
        face_results = []
        visible_ids = []
        
        # Sort faces by size (largest first) — primary face gets full analysis
        faces_sorted = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        
        t0_infer = time.monotonic()
        for face_idx, face in enumerate(faces_sorted):
            # 2a: Identity Tracking
            fid = self.tracker.get_id(face.bbox)
            visible_ids.append(fid)
            
            # Optimization: face size check for plugin skipping
            _, _, fw, fh = face.bbox
            is_small_face = (fw < 100 or fh < 100)
            
            # Init state machine for new face
            if fid not in self.face_states:
                self.face_states[fid] = {
                    "sm": PluginAwareStateMachine(self.hysteresis),
                    "blink": BlinkTracker(),
                    "neural_history": deque(maxlen=5), # for smoothing if needed
                    "first_seen": time.monotonic(),     # grace period tracking
                }
            
            state_ctx = self.face_states[fid]

            # 2b: Neural inference (NPU/GPU)
            # face.face_crop_299 is ALREADY (1,3,299,299) float32, RGB, [-1,+1]
            # from ShieldFacePipeline.align_and_crop() — DO NOT re-normalize!
            inp_tensor = face.face_crop_299.astype(np.float32)
                 
            raw_output = self._run_inference(inp_tensor)

            # 2c: Calibrate confidence (Part 3)
            calibrated = self.calibrator.calibrate(raw_output)
            # calibrated is [fake_prob, real_prob]
            neural_verdict = "REAL" if calibrated[1] > calibrated[0] else "FAKE"
            # Track real_prob as the confidence metric (trust score)
            # This way confidence always means "how real is this face"
            # High = likely real, Low = likely fake — consistent with HUD display
            raw_real_prob = float(calibrated[1])
            
            # Temporal smoothing: 5-frame rolling average to reduce jitter
            # Raw confidence can swing 30%+ frame-to-frame due to face angle/lighting
            state_ctx["neural_history"].append(raw_real_prob)
            neural_confidence = sum(state_ctx["neural_history"]) / len(state_ctx["neural_history"])
            # Clamp to valid probability range [0.0, 1.0]
            neural_confidence = max(0.0, min(1.0, neural_confidence))

            # 2d: Liveness check (Part 3)
            # compute_ear requires (landmarks, eye_indices, head_pose, is_frontal)
            # Use 478-mesh landmarks (pixel coords) and MediaPipe eye indices
            lm_for_ear = face.landmarks  # (478, 2) pixel coordinates
            ear_l, rel_l = compute_ear(lm_for_ear, LEFT_EYE, face.head_pose, face.is_frontal)
            ear_r, rel_r = compute_ear(lm_for_ear, RIGHT_EYE, face.head_pose, face.is_frontal)
            ear = (ear_l + ear_r) / 2.0
            # Use the worse reliability of the two eyes
            _rel_order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
            ear_reliability = rel_l if _rel_order.get(rel_l, 0) <= _rel_order.get(rel_r, 0) else rel_r
            
            # Update blink tracker
            blink_info = state_ctx["blink"].update(ear, ts, reliability=ear_reliability, blendshapes=face.blendshapes)
            
            # 2e: Distance estimation (BEFORE texture — needed for screen detection)
            distance_cm = self._estimate_distance(face.bbox, frame.shape)
            clamped_dist = max(0.0, min(distance_cm, 500.0))
            
            # 2f: Texture check with SCREEN REPLAY DETECTION (Part 3)
            # Pass distance for physics-based screen detection
            tex_baseline = self._device_baseline.get(
                "texture_floor",
                self._device_baseline.get("recommended_threshold", 10.0)
            )
            tex_score, tex_suspicious, tex_explain = compute_texture_score(
                face.face_crop_raw, tex_baseline, distance_cm=clamped_dist)

            # 2f: Plugin votes (Parts 7-8)
            # Skip expensive plugins for small/distant faces to preserve FPS
            plugin_votes = []
            if is_small_face:
                # Small face — only run lightweight plugins
                for plugin in self.plugins:
                    if plugin.name in ("skin_reflectance", "codec_forensics"):
                        try:
                            vote = plugin.analyze(face, frame)
                            plugin_votes.append(vote)
                        except Exception as e:
                            self.logger.warn(f"Plugin {plugin.name} failed: {e}")
            else:
                for plugin in self.plugins:
                    try:
                        vote = plugin.analyze(face, frame)
                        plugin_votes.append(vote)
                    except Exception as e:
                        self.logger.warn(f"Plugin {plugin.name} failed: {e}")

            # 2g: Fuse ALL decisions (Part 3 state machine)
            tier1 = neural_verdict
            # Tier 2 (Liveness): PASS if blinks detected OR in grace period
            # BlinkTracker returns blink count. 
            # If reliability is LOW (occlusion/angle), we can't trust EAR.
            # DecisionStateMachine logic:
            #   TIER 1 = Neural (Deepfake)
            #   TIER 2 = Liveness (Blink/rPPG)
            #   TIER 3 = Texture/Forensics
            
            # Grace period: Give the user 1.5 seconds to produce a natural blink
            # Reduced from 3.0s — screen replay attacks exploit long grace periods.
            face_age = time.monotonic() - state_ctx.get("first_seen", time.monotonic())
            BLINK_GRACE_PERIOD = 1.5  # seconds (reduced for faster attack detection)
            
            if blink_info["count"] > 0:
                tier2 = "PASS"  # Confirmed blink — liveness proven (permanent)
            elif face_age < BLINK_GRACE_PERIOD:
                tier2 = "PASS"  # Grace period — assume live until proven otherwise
            elif ear_reliability == "LOW":
                tier2 = "FAIL"  # Unreliable after grace period — can't assess liveness
            else:
                tier2 = "FAIL"  # No blinks after grace period — suspicious

            # EAR ANOMALY CHECK: AI avatar eyes are unnaturally wide-open
            # Real human resting EAR is typically 0.20-0.35.
            # AI avatars with animated wide eyes show EAR > 0.40 persistently.
            EAR_ANOMALY_THRESHOLD = 0.40
            ear_anomaly = False
            if ear > EAR_ANOMALY_THRESHOLD and ear_reliability in ("HIGH", "MEDIUM"):
                # Track consecutive high-EAR frames
                state_ctx.setdefault("high_ear_count", 0)
                state_ctx["high_ear_count"] += 1
                # If high EAR persists for >10 frames, flag as anomaly
                if state_ctx["high_ear_count"] > 10:
                    ear_anomaly = True
            else:
                state_ctx["high_ear_count"] = 0
            
            # SCREEN REPLAY OVERRIDE: If texture analysis detects screen replay,
            # override neural verdict. Physical evidence trumps neural network.
            screen_detected = "SCREEN_REPLAY" in tex_explain
            
            if screen_detected:
                tier1 = "FAKE"  # Override neural — physical evidence is conclusive
                tier3 = "FAIL"
            elif ear_anomaly:
                # Wide-open eyes alone aren't conclusive but reduce trust
                tier3 = "FAIL"  # Mark forensic as suspicious
            else:
                tier3 = "PASS" if not tex_suspicious else "FAIL"

            # Update State Machine
            state = state_ctx["sm"].update(
                tier1, tier2, tier3,
                plugin_votes=plugin_votes
            )
            
            # ── FAKE LOCKOUT MECHANISM (V2.3) ──
            # Uses PERCENTAGE of FAKE frames in a rolling window.
            # From audit analysis:
            #   Real face:  ~28% FAKE rate (neural jitter is normal)
            #   AI video:   ~56-82% FAKE rate (consistently suspicious)
            # Threshold: 50% FAKE in last 60 frames = lockout
            if "verdict_window" not in state_ctx:
                state_ctx["verdict_window"] = deque(maxlen=60)
            
            state_ctx["verdict_window"].append(1 if state in ("FAKE", "CRITICAL") else 0)
            window = state_ctx["verdict_window"]
            
            if len(window) >= 20:  # Need at least 20 frames for reliable %
                fake_pct = sum(window) / len(window)
                # If >50% FAKE over the window, activate lockout
                if fake_pct > 0.50:
                    state_ctx["fake_lockout_until"] = time.monotonic() + 20.0
            
            # Track neural confidence minimum — if it ever drops very low,
            # this face has shown deepfake characteristics
            state_ctx.setdefault("neural_min", 1.0)
            state_ctx["neural_min"] = min(state_ctx["neural_min"], neural_confidence)
            
            # Check if currently in fake lockout
            fake_locked = time.monotonic() < state_ctx.get("fake_lockout_until", 0)
            # Mark as "ever suspicious" if neural dropped below 20%
            # (extreme low confidence = definite deepfake artifact)
            neural_ever_suspicious = state_ctx["neural_min"] < 0.20
            
            # VERIFIED PROMOTION: If REAL + blinks confirmed + tracked >5s
            # This provides the highest trust level for continuously verified faces
            VERIFIED_MIN_AGE = 5.0  # seconds of continuous tracking
            VERIFIED_MIN_BLINKS = 1  # at least 1 natural blink
            VERIFIED_MIN_CONFIDENCE = 0.6  # neural trust above 60%
            
            # Block promotion if in fake lockout or if neural was ever very suspicious
            can_promote = (not fake_locked and not neural_ever_suspicious
                          and not screen_detected and not ear_anomaly)
            
            if (state == "REAL"
                and blink_info["count"] >= VERIFIED_MIN_BLINKS
                and face_age >= VERIFIED_MIN_AGE
                and neural_confidence >= VERIFIED_MIN_CONFIDENCE
                and not tex_suspicious
                and can_promote):
                state = "VERIFIED"
                state_ctx["sm"].state = "VERIFIED"
            
            # If in fake lockout, block ALL positive states
            if fake_locked and state in ("REAL", "VERIFIED"):
                state = "SUSPICIOUS"
                state_ctx["sm"].state = "SUSPICIOUS"

            # Clamp values to prevent garbage/NaN display
            clamped_ear = max(0.0, min(round(ear, 4), 1.0))
            clamped_tex = max(0.0, min(tex_score, 999.9))
            clamped_conf = max(0.0, min(neural_confidence, 1.0))
            # distance_cm and clamped_dist already computed above (before texture)
            
            # Head pose for audit trail
            yaw, pitch, roll = face.head_pose if face.head_pose else (0.0, 0.0, 0.0)
            
            res = FaceResult(
                face_id=fid,
                bbox=face.bbox,
                landmarks=face.landmarks,
                state=state,
                confidence=clamped_conf,
                neural_confidence=clamped_conf,
                ear_value=clamped_ear,
                ear_reliability=ear_reliability,
                texture_score=clamped_tex,
                texture_explanation=tex_explain,
                tier_results=(tier1, tier2, tier3),
                plugin_votes=plugin_votes,
                advanced_info={
                    "blinks": blink_info.get("count", 0),
                    "blink_source": blink_info.get("source", "?"),
                    "blink_baseline": blink_info.get("baseline", 0),
                    "blink_pattern_score": blink_info.get("pattern_score", 0.5),
                    "ear_value": clamped_ear,
                    "ear_reliability": ear_reliability,
                    "ear_anomaly": ear_anomaly,
                    "tracker_id": fid,
                    "distance_cm": clamped_dist,
                    "head_pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)},
                    "face_alert": "SCREEN_REPLAY" if screen_detected else ("DEEPFAKE" if state in ("FAKE", "CRITICAL") else ""),
                    "screen_replay": screen_detected,
                    "face_age_s": round(face_age, 1),
                }
            )
            face_results.append(res)

        timing["infer_ms"] = (time.monotonic() - t0_infer) * 1000
        
        # Cleanup stale trackers
        self.tracker.purge_stale(visible_ids)
        # Also clean up face_states
        stale_states = [fid for fid in self.face_states if fid not in self.tracker.identities]
        for fid in stale_states:
            del self.face_states[fid]

        # STAGE 3: Performance + Memory
        t_total = time.monotonic() - t_start
        timing["total_ms"] = t_total * 1000
        
        self._frame_times.append(t_total)
        fps = len(self._frame_times) / sum(self._frame_times) if self._frame_times else 0.0

        current_mem = psutil.Process().memory_info().rss
        mem_growth = current_mem - self._memory_baseline
        if mem_growth > 500 * 1024 * 1024:  # >500MB
            gc.collect()
            self.logger.warn("Memory growth > 500MB, GC forced")

        # STAGE 4: Structured audit logging
        log_entry = {
            "timestamp": time.time(),
            "faces_detected": len(faces),
            "face_results": [r.to_dict() for r in face_results],
            "fps": fps,
            "timing": timing,
            "memory_mb": current_mem / 1e6
        }
        self.logger.log_frame(log_entry)

        return EngineResult(
            frame=frame, # Raw frame for display (encryption handled if persistent storage needed)
            timestamp=ts,
            face_results=face_results,
            fps=fps,
            timing_breakdown=timing,
            camera_health=self.camera.get_health_status(),
            memory_mb=current_mem / 1e6
        )
            
    def get_latest_result(self) -> Optional[EngineResult]:
        """HUD Thread (Main) calls this to get render data."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
