# Shield-Ryzen V2 Changelog

## V2.3.0 (The Precision Update - LATEST)
### Critical Fixes
*   **Eliminated False Positives on Real Faces**: Physics thresholds recalibrated using audit data — real faces at 50cm (texture 200-600) no longer falsely trigger SCREEN_REPLAY.
*   **Percentage-Based FAKE Lockout**: Replaced raw count lockout with rolling 60-frame window analysis. Lockout triggers only when FAKE % exceeds 50% (AI videos = 56-82%, real faces = 28%).
*   **Calibrated Screen Detection Layers**: Moiré threshold 0.25→0.50, Screen Light 0.30→0.50, Signal Fusion requires 3+ signals instead of 2.

### New Features
*   **Real-Time Texture Debug Log**: `logs/texture_debug.log` captures raw moire/light/physics scores every frame for live analysis.
*   **Audit-Driven Threshold Tuning**: All thresholds validated against 24 tracked face identities and 21,000+ frames of real-world data.

## V2.2.0 (The Anti-Replay Update)
### Security Features
*   **5-Layer Screen Detection**: Physics-based distance-texture cross-validation + Moiré grid analysis + Screen Light Emission detection.
*   **Neural Trace Memory**: Tracks historical confidence minima to prevent deepfake "recovery" after detection.
*   **Flash Alerts**: Visual flashing red "!! SCREEN REPLAY ATTACK !!" overlay in HUD.

## V2.1.0 (Unified Release - NPU Optimized)
### New Features
*   **Ryzen AI Engine**: `v3_xdna_engine.py` with Vitis AI Execution Provider priority.
*   **Shield HUD**: Complete WCAG 2.1 AA compliant overlay (`shield_hud.py`).
*   **Enterprise Security**: `plugins/arcface_reid.py` + `security/audit_trail.py` (SHA-256 Chained Logs).
*   **Hardware Monitor**: `shield_hardware_monitor.py` tracking CPU/RAM/FPS/Power.
*   **Validation Suite**: `validate_system.py` + `benchmarks/` + `edge_case_test_suite.py`.

### Improvements
*   **Launcher**: Unified `shield.py` (`start.bat`) for easy execution.
*   **Documentation**: Verified Claims vs Evidence, updated Architecture.
*   **Quantization**: `quantize_ryzen.py` for INT8 QDQ export.

---

## V2.0.0 (Shield-Ryzen V2)
### Major Architecture
*   **Modular Architecture**: Replaced the monolithic `shield_xception.py` with specific modules:
    *   `shield_camera.py`: Modular camera handling.
    *   `shield_face_pipeline.py`: Dedicated face detection logic.
    *   `shield_utils_core.py`: Shared utilities and math functions.
*   **Engine Types**: Introduced multiple engine variants:
    *   `v2_onnx.py`: Basic FP32 ONNX runtime.
    *   `v3_int8_engine.py`: INT8 Quantized engine (Standard).
    *   `v3_xdna_engine.py`: Experimental AMD XDNA engine (`.xmodel`).
*   **Configuration**: All thresholds now configurable via `config.yaml` and `decision_thresholds.yaml`.

### New Features (Part 7, 8, 9)
*   **Accessible HUD**: Completely rewritten visual feedback with shape indicators and audio cues.
*   **Temporal Smoothing**: Hysteresis-based state machine eliminates flickering decisions.
*   **Adaptive Thresholding**: Laplacian texture threshold adapts to device baseline.
*   **EAR Compensation**: Eye Aspect Ratio (EAR) now compensates for head pose angles.
*   **Forensics**:
    *   **Temporal Consistency**: Detects lighting/texture changes over time.
    *   **Frequency Analysis**: Detects GAN spectral artifacts.
    *   **Lip-Sync**: Checks for audio-visual correlation.
    *   **Deepfake Attribution**: Classifies generator type (e.g., Face2Face).
*   **AMD Hardware Support**:
    *   XDNA overlay compilation pipeline (`compile_xmodel.sh`).
    *   Zero-copy memory buffer allocation.
    *   Direct camera capture via AMD SDKs.
    *   fTPM-backed model integrity verification.

### Compliance & Verification (Part 10, 11)
*   **GDPR/BIPA Compliance**: Documented privacy processes and local-only processing guarantee.
*   **Benchmark Suite**: Comprehensive `benchmarks/` covering FPS, accuracy, power, and thresholds.
*   **Evidence Package**: Reproducible build artifacts consolidatd in `evidence_package/`.
*   **Unit Tests**: Achieved **73/73 PASSED** tests covering all modules.

### Bug Fixes
*   Fixed 40+ legacy bugs including hardcoded paths, incorrect tensor shapes, missing dependencies, and race conditions in logging.
*   Corrected EAR logic to handle non-frontal faces gracefully.
*   Ensured thread-safety in HUD rendering.
*   Patched memory leaks in long-running inference loops.

### Known Limitations
*   **NPU Availability**: XDNA acceleration falls back to CPU cleanly but was not available on the test machine (0% coverage).
*   **Power Metrics**: Battery drain could not be measured on AC power without external sensors.
*   **Adversarial Robustness**: Not tested against specific adversarial attacks (white-box).

---

## V1.0.0 (Legacy)
*   Initial prototype. High claims, low verification.
*   Hardcoded logic and deeply nested monolithic script.
*   "Military-grade" marketing language (Deprecated).
