
# Shield-Ryzen V2 Changelog

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
