# Shield-Ryzen V2 Architecture

## Overview
Shield-Ryzen V2 is a real-time deepfake detection system optimized for AMD Ryzen AI NPUs. It employs a multi-layered defense strategy combining neural analysis, signal forensics, and biometric verification.

## 🛡️ Anti-Replay Architecture (Physics Shield)
The V2.2 update introduces a 5-layer physical verification system designed to catch screen-replay attacks (showing a phone/tablet to the webcam).

| Layer | Method | Goal |
|---|---|---|
| **Layer 1** | **Adaptive Laplacian** | Standard sharpness/blur check using device-specific baseline. |
| **Layer 2** | **Physics Cross-Validation** | Enforces inverse-square law: max texture sharpness $T \propto 1/D^2$. Screens violate this by being too sharp at distance. |
| **Layer 3** | **Moiré Grid Analysis** | Detects periodic high-frequency patterns caused by screen pixel grid interference. |
| **Layer 4** | **Light Emission** | Detects backlighting signatures (blue-shift, uniform brightness variance, narrow chrominance gamut). |
| **Layer 5** | **Signal Fusion** | Combines weak signals from multiple layers to perform a majority-rule forensic verdict. |


## Core Components

### 1. Engine (`shield_engine.py`)
- **Architecture**: Asynchronous Triple-Buffer Pipeline.
  - `_cam_thread`: High-speed capture (OpenCV/DirectShow).
  - `_ai_thread`: Neural inference & plugin analysis.
  - Main Thread: HUD rendering & User Interaction.
- **State Machine**: `IdentityTracker` maintains per-face trust scores across frames.
- **Optimization**: `RyzenXDNAEngine` subclass (`v3_xdna_engine.py`) enforces NPU offloading via Vitis AI EP.

### 2. Neural Network (`shield_xception.py` / `shield_ryzen_int8.onnx`)
- **Backbone**: XceptionNet (pretrained on FaceForensics++ c23).
- **Format**: ONNX INT8 (Quantized-DeQuantized) for NPU.
- **Preprocessing**: 299x299 RGB, Deviation Normalization via `ShieldFacePipeline`.

### 3. Plugin System (`shield_plugin.py`)
Modular security layers executed post-inference:

| Category | Plugin Class | File | Logic |
|---|---|---|---|
| **Biometric** | `BlinkPlugin` | `plugins/blink_detector.py` | EAR < 0.2 (Temporal) |
| | `HeartbeatPlugin` | `plugins/heartbeat_monitor.py` | rPPG (Green ch. FFT) |
| **Forensic** | `FrequencyAnalyzer` | `plugins/frequency_analyzer.py` | HFER (High-Freq Energy Ratio) |
| | `CodecForensics` | `plugins/codec_forensics.py` | 8x8 Block Gradients |
| | `AdversarialDetector` | `plugins/adversarial_detector.py` | Gradient Clustering |
| **Enterprise** | `ArcFacePlugin` | `plugins/arcface_reid.py` | Cosine Similarity (Encrypted DB) |
| **Interactive** | `LipSyncPlugin` | `plugins/lip_sync_verifier.py` | Phoneme Challenge-Response |

### 4. User Interface (`shield_hud.py`)
- **Rendering**: Low-latency OpenCV overlay (< 5ms).
- **Accessibility**: WCAG 2.1 AA compliant (Shapes + Colors).
- **Features**: Real-time confidence badges, Tech Detail overlay, Face Mesh wireframe.

### 5. Security & Compliance
- **Cryptography**: `shield_crypto.py` (AES-256 for biometric data).
- **Audit Trail**: `security/audit_trail.py` (SHA-256 hash-chained logs).
- **Compliance**: Adheres to GDPR Art 9, Illinois BIPA, and EU AI Act (See `docs/COMPLIANCE.md`).

## Data Flow
1. **Input**: Camera Frame -> `ShieldCamera` (30 FPS).
2. **Detection**: MediaPipe Face Detection -> `ShieldFacePipeline`.
3. **Alignment**: 5-Point alignment -> 299x299 Crop.
4. **Inference**: ONNX Runtime (NPU) -> Neural Confidence.
5. **Analysis**: `PluginManager` aggregates plugin votes.
    - If `Heartbeat` fails -> Reduce score.
    - If `Frequency` anomalous -> Flag FAKE.
6. **Decision**: `PluginAwareStateMachine` updates persistent face state (REAL/FAKE/SUSPICIOUS).
7. **Output**: HUD Overlay + Audio Alert + Audit Log.

## Hardware Optimization
- **Ryzen AI**: Leveraging Vitis AI EP for INT8 acceleration.
- **Zero-Copy**: Planned for V2.1 via `OrtValue`.
- **Telemetry**: `ShieldHardwareMonitor` tracks power/thermal estimation.

## Directory Structure
- `shield_*.py`: Core logic files.
- `plugins/`: Security modules.
- `security/`: Audit & Crypto.
- `models/`: ONNX/PTH weights.
- `tests/`: Validation suite.
- `docs/`: Compliance & Architecture documentation.
