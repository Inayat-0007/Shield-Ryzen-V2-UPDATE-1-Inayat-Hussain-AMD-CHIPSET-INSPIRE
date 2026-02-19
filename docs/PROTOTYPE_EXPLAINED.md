# 🛡️ Shield-Ryzen V2 — Complete Prototype Explanation
### Developer: Inayat Hussain | AMD Slingshot 2026
### Generated: 2026-02-20

---

## 📌 What Is This Prototype?

Shield-Ryzen V2 is a **real-time deepfake detection system** built for the **AMD Slingshot 2026 competition** by Inayat Hussain. It uses a **triple-layered security approach** — combining deep learning, biometric verification, and forensic analysis — to determine if a live webcam face is a **real human** or a **deepfake/replay/mask attack**. Everything runs **100% locally on-device** (no cloud), targeting AMD Ryzen AI NPU hardware.

---

## 🏗️ Architecture Overview — The 14-Part System

The prototype is organized into **14 modular parts**, each a separate `.py` file with a clear responsibility:

```
┌──────────────────────────────────────────────────────────────────┐
│                     start_shield.py (PART 14)                    │
│                        ENTRY POINT / LAUNCHER                    │
└──────────────┬───────────────────────────────────┬───────────────┘
               │ Creates & starts                  │ Creates
               ▼                                   ▼
┌──────────────────────────┐          ┌──────────────────────┐
│   shield_engine.py       │          │   shield_hud.py      │
│   (PART 6) — THE BRAIN   │◄─────────│   (PART 10) — HUD    │
│   Central Orchestrator   │ results  │   Display Overlay    │
└──────┬──────┬──────┬─────┘          └──────────────────────┘
       │      │      │
       │      │      └──── Plugins (PARTS 7-8)
       │      │              ├── rppg_heartbeat.py
       │      │              ├── stereo_depth.py
       │      │              ├── skin_reflectance.py
       │      │              ├── frequency_analyzer.py
       │      │              ├── codec_forensics.py
       │      │              ├── adversarial_detector.py
       │      │              ├── lip_sync_verifier.py
       │      │              ├── challenge_response.py
       │      │              └── arcface_reid.py
       │      │
       │      └──── shield_face_pipeline.py (PART 2)
       │              Face Detection & Preprocessing
       │
       └──── shield_camera.py (PART 1)
               Camera Capture & Validation
```

### Supporting Modules:

| File                        | Part     | Role                                          |
|-----------------------------|----------|-----------------------------------------------|
| shield_xception.py          | Part 4   | Core XceptionNet neural model                 |
| shield_utils_core.py        | Part 3   | EAR, Texture, BlinkTracker, StateMachine, Calibration |
| shield_crypto.py            | Part 6.3 | AES-256 biometric encryption                  |
| shield_logger.py            | Part 6.5 | JSONL structured audit logging                |
| shield_plugin.py            | Part 6.2 | Plugin abstract interface                     |
| shield_gradcam.py           | Part 10.2| Visual explainability heatmaps                |
| shield_hardware_monitor.py  | —        | System resource monitoring                    |
| v3_int8_engine.py           | Part 5   | INT8 quantized ONNX engine                    |
| v3_xdna_engine.py           | Part 9   | AMD XDNA NPU engine                           |

---

## 🔄 Complete Execution Flow (Step-by-Step)

Here is exactly what happens from the moment you run the system to when you see the verdict on screen:

### Step 1: Launch (start_shield.py)

```
python start_shield.py --source 0 --model models/shield_ryzen_int8.onnx
```

1. Parses CLI arguments (camera source, model path, CPU/NPU mode, audit toggle)
2. Selects engine class: RyzenXDNAEngine (NPU) or ShieldEngine (CPU/GPU)
3. Creates ShieldHUD for display
4. Creates the chosen engine with config
5. Calls engine.start() → launches async threads
6. Enters the main loop: polls engine.get_latest_result() → passes to hud.render() → shows in cv2.imshow window

### Step 2: Engine Initialization (shield_engine.py → ShieldEngine.__init__)

When the engine is created, it initializes ALL subsystems in order:

1. Logger — Opens logs/shield_audit.jsonl for structured audit trail
2. Camera — ShieldCamera(camera_id=0, 640×480) opens the webcam
3. Face Pipeline — ShieldFacePipeline(mediapipe) loads MediaPipe FaceLandmarker model
4. Model — Loads the ONNX model (shield_ryzen_int8.onnx) or PyTorch weights (ffpp_c23.pth)
5. Confidence Calibrator — Temperature scaling (T=1.5) to fix overconfident softmax
6. Identity Tracker — IOU-based face tracking across frames
7. Plugins — Registers all enabled detection plugins (Heartbeat, Skin Reflectance, Frequency, Codec, Adversarial, etc.)
8. Crypto — AES-256 ephemeral key generated for memory encryption
9. Calibration — Loads or creates device baseline from shield_calibration.json

### Step 3: Triple-Buffer Async Pipeline (3 concurrent threads)

Once engine.start() is called, three threads run in parallel:

```
┌───────────────┐     Queue(2)     ┌───────────────┐     Queue(2)     ┌───────────────┐
│  THREAD 1     │ ──────────────► │  THREAD 2     │ ──────────────► │  MAIN THREAD  │
│  Camera       │  (frame, ts)    │  AI/Analysis  │  EngineResult   │  HUD/Display  │
│  Capture      │                 │  Processing   │                 │  Rendering    │
└───────────────┘                 └───────────────┘                 └───────────────┘
```

- Thread 1 (Camera): Continuously calls camera.read_validated_frame() and puts valid frames into camera_queue
- Thread 2 (AI): Takes frames from camera_queue, runs the full analysis pipeline (_process_frame), puts results into result_queue
- Main Thread (HUD): Takes results from result_queue, renders the overlay via ShieldHUD, displays with OpenCV

Why triple-buffer? Camera capture and AI inference run at different speeds. The queues (max size 2) ensure the system always shows the latest frame, dropping stale ones to prevent lag.

### Step 4: Camera Capture & Validation (shield_camera.py)

ShieldCamera wraps cv2.VideoCapture with comprehensive validation:

1. Backend Selection: Uses DirectShow (Windows) to avoid MSMF buffering
2. 1-frame buffer: CAP_PROP_BUFFERSIZE=1 to minimize latency
3. Validation Checklist for every frame:
   - ret == True (capture succeeded)
   - Frame is numpy.ndarray
   - Shape is 3D (H, W, 3) — three color channels
   - dtype == uint8
   - Frame is not all-black (brightness > threshold)
   - Frame is not frozen (differs from previous frame)
4. Freshness Check: check_frame_freshness() rejects frames older than 500ms
5. Health Monitoring: Tracks FPS, drop count, lag spikes, resolution

### Step 5: Face Detection & Preprocessing (shield_face_pipeline.py)

The ShieldFacePipeline handles ALL face-related processing:

1. Detection via MediaPipe FaceLandmarker:
   - Detects up to 4 faces per frame
   - Returns 478 face mesh landmarks per face
   - Extracts blendshape coefficients (used for blink detection)

2. Landmark Conversion:
   - Converts 478-point mesh → standard 68-point format
   - Uses anatomically-documented mapping indices

3. Head Pose Estimation:
   - Uses cv2.solvePnP with 6 key landmarks matched to a 3D generic face model
   - Returns (yaw, pitch, roll) in degrees
   - Determines if face is frontal (important for EAR reliability)

4. Occlusion Estimation:
   - Checks if eyes, nose, mouth, forehead landmarks are within the bounding box
   - If occlusion > 0.5, EAR results are flagged unreliable

5. Face Alignment & Cropping:
   - Aligns face using eye center positions
   - Crops to 299×299 pixels (Xception input requirement)
   - Normalizes: (pixel/255 - 0.5) / 0.5 → maps [0,255] to [-1.0, +1.0]
   - Converts BGR→RGB
   - Returns (1, 3, 299, 299) NCHW tensor ready for inference

### Step 6: The 3-Tier Security Decision (_process_frame in shield_engine.py)

For EACH detected face, the engine runs three independent verification tiers:

#### TIER 1 — Neural Network (XceptionNet)

```
Face Crop (299×299) → XceptionNet → Softmax → [fake_prob, real_prob]
                                        ↓
                              Temperature Scaling (T=1.5)
                                        ↓
                              calibrated [fake_prob, real_prob]
```

- Model: XceptionNet (via timm.legacy_xception with 2 output classes)
- Weights: ffpp_c23.pth — trained on FaceForensics++ dataset (compressed, quality c23)
- Output: [fake_prob, real_prob] after softmax
- Temperature Scaling: Raw softmax is overconfident (outputs 0.99 when should be 0.7).
  The ConfidenceCalibrator applies: logits = log(probs) → scaled_logits = logits / T → re-softmax
- Verdict: REAL if real_prob > fake_prob, else FAKE
- ONNX variant: Same model exported to ONNX format, with INT8 quantization for AMD NPU

#### TIER 2 — Liveness (Blink Detection)

```
Face Landmarks → compute_ear() → BlinkTracker.update()
                     ↓                    ↓
           Eye Aspect Ratio      Blink Count, Pattern Score
```

- EAR (Eye Aspect Ratio): Formula = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
  - Normal open eye: ~0.25–0.30
  - Blink (closed): ~0.15–0.18
  - Cosine angle compensation: If head is turned, raw EAR changes. The system corrects
    for yaw via cos(yaw_radians) scaling
  - Reliability tiers: HIGH (frontal + low occlusion), MEDIUM, LOW (turned/occluded)

- BlinkTracker uses Dynamic Baseline Scaling (DBS):
  - Learns YOUR baseline EAR during runtime (adapts to individual eye openness)
  - Uses blendshapes (MediaPipe) as primary signal (higher precision)
  - Falls back to geometric EAR if blendshapes unavailable
  - Detects blink when: signal drops below baseline × 0.75 and stays for 50-400ms
  - Scores blink pattern naturalness: 0.0 (robotic/regular) → 1.0 (natural/irregular)

- Tier 2 Verdict: PASS if blink count > 0 (real human blinks), FAIL otherwise

#### TIER 3 — Forensic (Texture Analysis)

```
Face Crop (raw BGR) → Forehead ROI extraction → Laplacian Variance + FFT
                                                        ↓
                                              (score, is_suspicious, explanation)
```

- Laplacian Variance: Measures edge sharpness in the forehead region
  - Real faces: Lots of micro-detail (pores, wrinkles) → high variance
  - Deepfakes/Screens: Smoothed/blurred → low variance
  - Threshold: 15 (configurable)

- FFT High-Frequency Energy: Computes the ratio of high-frequency vs total energy
  - Real faces have balanced spectrum
  - GANs suppress high frequencies (smoothness artifact)
  - Screens have Moiré patterns (unusual HF spikes)

- Tier 3 Verdict: PASS if texture looks natural, FAIL if suspicious

### Step 7: Decision State Machine Fusion (DecisionStateMachine)

All three tiers are fused using a truth table with temporal hysteresis:

| Tier 1 (Neural) | Tier 2 (Liveness) | Tier 3 (Forensic) | → Final State       |
|------------------|--------------------|--------------------|----------------------|
| REAL             | PASS               | PASS               | ✅ VERIFIED          |
| REAL             | FAIL               | PASS               | ⏳ WAIT_BLINK        |
| REAL             | PASS               | FAIL               | ⚠️ SUSPICIOUS        |
| FAKE             | any                | any                | 🔴 FAKE / HIGH_RISK  |
| REAL             | FAIL               | FAIL               | ⚠️ SUSPICIOUS        |

Temporal Hysteresis: The state machine requires the same verdict for 5 consecutive frames
before transitioning (prevents flickering). This is controlled by hysteresis_frames config.

Plugin-Aware Extension: The PluginAwareStateMachine downgrades Tier 3 to FAIL if any plugin
votes FAKE, ensuring forensic plugins influence the final decision.

### Step 8: Plugin Analysis (Parts 7-8)

Each registered plugin runs analyze(face_data, frame) and returns a structured vote:

| Plugin                     | Tier      | What It Detects                                                    |
|----------------------------|-----------|--------------------------------------------------------------------|
| HeartbeatPlugin (rPPG)     | Biometric | Remote pulse detection via green-channel FFT. 40-180 BPM = real.   |
| SkinReflectancePlugin      | Biometric | Skin light reflectance (real skin vs. printed/screen surface)      |
| StereoDepthPlugin          | Biometric | 3D depth from stereo camera (flat screen = no depth variation)     |
| ChallengeResponsePlugin    | Biometric | Interactive challenges (turn head, blink on command)               |
| FrequencyAnalyzerPlugin    | Forensic  | 2D FFT frequency fingerprint — GANs suppress high-freq details     |
| CodecForensicsPlugin       | Forensic  | Compression artifact patterns (double-encoded video = deepfake)    |
| AdversarialPatchPlugin     | Forensic  | Physical adversarial patches (glasses, stickers with patterns)     |
| LipSyncPlugin              | Forensic  | Lip-audio synchronization verification                             |

Each plugin returns:
```python
{
    "verdict": "REAL" | "FAKE" | "UNCERTAIN",
    "confidence": 0.0-1.0,
    "explanation": "Human-readable reason",
    "metric_value": float  # Raw measurement
}
```

### Step 9: Identity Tracking (IdentityTracker)

Faces need to be tracked across frames so that state machines, blink counters, and signal
buffers persist for the same person:

- Uses IOU (Intersection Over Union) matching
- If a new face's bounding box overlaps > 30% with a known face, it's the same person
- If no match → new identity (new state machine, blink tracker, etc.)
- Stale identities (not seen for 30 frames) are purged along with their state

### Step 10: HUD Rendering (shield_hud.py)

The ShieldHUD creates a WCAG 2.1 AA accessible overlay:

- Per-face bounding box: Corner bracket style with state-specific colors
- Color coding:
  - 🟢 Green (VERIFIED/REAL) — checkmark shape
  - 🟠 Orange (SUSPICIOUS) — question mark shape
  - 🔴 Red (FAKE/HIGH_RISK) — X mark shape
  - ⚫ Gray (UNKNOWN) — dash shape
- Label: Shows state + confidence percentage
- Technical details: Neural confidence, EAR value, texture score below each face
- Status bar: Top of screen shows "SHIELD-RYZEN V2", FPS, memory usage

Why shapes + colors? For color-blind accessibility — each state has both a unique color
AND a unique shape.

---

## 🔐 Security Features

### Model Integrity (shield_xception.py)
- SHA-256 hash verification of model weights at load time
- Raises ModelTamperingError if the hash doesn't match
- Shape verification for ONNX models (input must be [1,3,299,299])
- Expected key count verification (276 keys for Xception)

### Biometric Encryption (shield_crypto.py)
- AES-256-GCM encryption for in-memory biometric data
- Ephemeral keys — generated per session, never stored to disk
- Secure wipe — keys are dereferenced on shutdown
- Prevents RAM dump attacks from extracting face data

### Audit Trail (shield_logger.py)
- Every frame processed is logged to logs/shield_audit.jsonl
- Each log entry contains: timestamp, face count, per-face verdicts, FPS, timing breakdown, memory usage
- Thread-safe writes with locking
- Structured JSONL format for post-mortem analysis

---

## ⚡ Performance Architecture

### ONNX Runtime + INT8 Quantization (Parts 5, 9)
- Model exported from PyTorch to ONNX format
- INT8 static quantization reduces model size from ~83MB → ~21MB
- Execution provider priority: VitisAIExecutionProvider (AMD NPU) → CUDAExecutionProvider (NVIDIA) → CPUExecutionProvider

### AMD XDNA Engine (v3_xdna_engine.py)
- Extends ShieldEngine for AMD Ryzen AI NPUs
- Enforces Vitis AI execution provider
- Boosts process priority on Windows (ABOVE_NORMAL_PRIORITY_CLASS)
- Ready for zero-copy input buffers when NPU supports it

### Memory Management
- Monitors RSS memory via psutil
- If growth exceeds 500MB, forces garbage collection
- Memory reported in every frame result

---

## 📁 Data Flow Summary

```
User's Face (webcam)
       │
       ▼
  ShieldCamera ───────── validates (shape, dtype, brightness, freshness)
       │
       ▼
  ShieldFacePipeline ─── detects faces, extracts 478 landmarks,
       │                  estimates head pose, aligns/crops to 299×299
       │
       ├──► TIER 1: XceptionNet inference → [fake_prob, real_prob]
       │         └─── Temperature scaling calibration
       │
       ├──► TIER 2: compute_ear() → BlinkTracker → blink count + pattern
       │
       ├──► TIER 3: compute_texture_score() → Laplacian + FFT
       │
       ├──► PLUGINS: Heartbeat(rPPG), FrequencyFFT, Adversarial, etc.
       │
       ▼
  DecisionStateMachine ── truth table fusion + 5-frame hysteresis
       │
       ▼
  FaceResult (state: VERIFIED/SUSPICIOUS/FAKE/UNKNOWN...)
       │
       ▼
  ShieldHUD ───────────── renders colored bounding boxes + badges
       │
       ▼
  cv2.imshow() ────────── displayed to user
       │
       ▼
  ShieldLogger ────────── every decision logged to JSONL
```

---

## 🧮 Key Mathematical Formulas

| Formula                                                | Where Used                  |
|--------------------------------------------------------|-----------------------------|
| EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)           | Eye blink detection         |
| EAR_corrected = EAR / cos(yaw_radians)                 | Head angle compensation     |
| Trust = 1 - raw_model_output                            | UX inversion (1.0=trusted)  |
| T(x) = softmax(log(x) / T)                             | Temperature scaling         |
| HFER = HF_energy / total_energy                         | FFT forensic analysis       |
| IOU = intersection_area / union_area                    | Face identity tracking      |
| BPM = peak_frequency × 60                              | rPPG heartbeat estimation   |
| Laplacian_var = var(∇²I)                                | Texture sharpness scoring   |

---

## 🔗 Component Connection Map

Here's how every file connects to every other file:

```
start_shield.py
  └─ imports: v3_xdna_engine (RyzenXDNAEngine, ShieldEngine), shield_hud (ShieldHUD)

v3_xdna_engine.py
  └─ inherits: shield_engine.ShieldEngine

shield_engine.py
  ├─ uses: shield_camera.ShieldCamera
  ├─ uses: shield_face_pipeline.ShieldFacePipeline
  ├─ uses: shield_xception.load_model_with_verification, ShieldXception
  ├─ uses: shield_utils_core.{ConfidenceCalibrator, DecisionStateMachine,
  │        compute_ear, compute_texture_score, BlinkTracker}
  ├─ uses: shield_crypto.{encrypt, decrypt, secure_wipe}
  ├─ uses: shield_plugin.ShieldPlugin
  ├─ uses: shield_logger.get_logger
  └─ uses: plugins/*.py (8 plugin files)

shield_face_pipeline.py
  └─ standalone (uses MediaPipe, OpenCV internally)

shield_camera.py
  └─ standalone (wraps cv2.VideoCapture)

shield_utils_core.py
  └─ standalone (pure math + state logic)

shield_hud.py
  └─ imports: shield_engine.{EngineResult, FaceResult}

All plugins/*.py
  └─ implement: shield_plugin.ShieldPlugin interface
```

---

## 📦 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.13                         |
| DNN Framework    | PyTorch + timm                      |
| Face Detection   | MediaPipe FaceLandmarker            |
| Vision I/O       | OpenCV                              |
| GPU Runtime      | CUDA (NVIDIA RTX 3050)              |
| Future Runtime   | ONNX Runtime (AMD Ryzen AI NPU)     |
| Inference Mode   | Real-time webcam (30 FPS target)    |
| Encryption       | AES-256-GCM (cryptography library)  |
| Logging          | JSONL structured audit              |
| Quantization     | INT8 static (ONNX Runtime)          |

---

## 📄 File Inventory (Complete)

### Core Engine Files
- start_shield.py        — Entry point / launcher (Part 14)
- shield_engine.py       — Central orchestrator / brain (Part 6)
- shield_camera.py       — Camera capture & validation (Part 1)
- shield_face_pipeline.py — Face detection & preprocessing (Part 2)
- shield_xception.py     — XceptionNet model wrapper (Part 4)
- shield_utils_core.py   — Shared security logic (Part 3)
- shield_hud.py          — HUD display overlay (Part 10)

### Security & Infrastructure
- shield_crypto.py       — AES-256 biometric encryption (Part 6.3)
- shield_logger.py       — JSONL audit logger (Part 6.5)
- shield_plugin.py       — Plugin interface (Part 6.2)
- shield_gradcam.py      — GradCAM explainability (Part 10.2)
- shield_hardware_monitor.py — System monitoring

### Engine Variants
- v3_int8_engine.py      — INT8 quantized engine (Part 5)
- v3_xdna_engine.py      — AMD XDNA NPU engine (Part 9)

### Plugins (plugins/ directory)
- rppg_heartbeat.py      — Remote pulse detection (Part 7.2)
- stereo_depth.py        — Stereo depth estimation (Part 7.3)
- skin_reflectance.py    — Skin reflectance analysis (Part 7.4)
- challenge_response.py  — Interactive challenges (Part 7.1)
- frequency_analyzer.py  — FFT forensic analysis (Part 8.1)
- codec_forensics.py     — Compression forensics (Part 8.2)
- adversarial_detector.py — Adversarial patch detection (Part 8.3)
- lip_sync_verifier.py   — Lip sync verification (Part 8.4)
- arcface_reid.py        — Face re-identification

### Model Files
- ffpp_c23.pth                     — PyTorch Xception weights (~83MB)
- shield_ryzen_v2.onnx             — FP32 ONNX model (~83MB)
- shield_ryzen_int8.onnx           — INT8 quantized ONNX model (~21MB)
- face_landmarker.task             — MediaPipe FaceLandmarker model
- face_landmarker_v2_with_blendshapes.task — MediaPipe model with blendshapes

### Configuration
- config.yaml            — Tunable security parameters
- shield_calibration.json — Device-specific calibration data
- requirements.txt       — Python dependencies

### Documentation
- README.md              — Project overview
- GEMINI.md              — AI agent workspace rules
- MODEL_CARD.md          — Model documentation
- CHANGELOG.md           — Version history
- SECURITY.md            — Security policy
- TRANSFER_GUIDE.md      — Porting guide
- CONTRIBUTING.md        — Contribution guidelines

---

End of Document.
