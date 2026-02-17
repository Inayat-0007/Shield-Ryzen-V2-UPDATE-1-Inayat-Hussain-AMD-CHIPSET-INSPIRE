# 🛡️ Shield-Xception Architecture Reference

## Overview
Shield-Xception is a **real-time deepfake detection system** built for the **AMD Slingshot 2026** competition. It uses an XceptionNet backbone trained on FaceForensics++ (c23 compression) to classify faces as Real or Fake via a live webcam feed, optimized through INT8 quantization for AMD Ryzen AI NPU deployment.

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webcam    │────▶│  MediaPipe   │────▶│  XceptionNet │────▶│  Security   │
│  (OpenCV)   │     │  FaceLandmark│     │  INT8 ONNX   │     │  Mode UI    │
│  30 FPS     │     │  478-pt mesh │     │  Softmax Out │     │  3-Tier     │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
       │                                        │
       │                  CUDA                  │
       └──────────── RTX 3050 GPU ──────────────┘
                      (AMD NPU Target)
```

## Data Flow (Per Frame)

1. **Capture:** OpenCV reads BGR frame from webcam (`cv2.VideoCapture(0)`)
2. **Detection:** Frame converted to RGB → MediaPipe FaceLandmarker extracts 478-point mesh
3. **Blink Detection:** EAR (Eye Aspect Ratio) calculated from eye landmarks
4. **Crop:** Each face cropped from original BGR frame using landmark-derived bounding box
5. **Texture Guard:** Laplacian variance check for smoothness artifacts
6. **Transform:** Crop → Resize 299×299 → Float32 → Normalize [-1, 1] → NCHW
7. **Inference:** Input → ONNX Runtime (INT8 QDQ) → Softmax → `[Fake, Real]` probabilities
8. **Classification:** 3-Tier Security Mode decision based on probabilities + liveness + texture
9. **Display:** Bounding box + label overlay on original frame → `cv2.imshow()`

## ShieldXception Model Architecture

```python
ShieldXception(nn.Module)
├── self.model = timm.create_model('legacy_xception', pretrained=False, num_classes=2)
│   ├── Entry Flow (3 conv blocks with separable convolutions)
│   ├── Middle Flow (8 repeated blocks)
│   ├── Exit Flow (2 blocks + global average pooling)
│   └── FC Head → 2 output neurons
└── forward: logits → torch.softmax(dim=1)  # Output sums to 1.0
```

- **Input:** `[B, 3, 299, 299]` — Batch of RGB face crops
- **Output:** `[B, 2]` — Softmax probabilities `[Fake, Real]`
  - Index 0 = Fake probability
  - Index 1 = Real probability

## Security Mode Classification

| Priority | Condition | Label | Color |
|---|---|---|---|
| 1 | `fake_prob > 0.50` | CRITICAL: FAKE DETECTED | 🔴 Red |
| 2 | `real_prob < 0.89` | WARNING: LOW CONFIDENCE | 🟡 Yellow |
| 3 | No blink in 10s | LIVENESS FAILED | 🟠 Orange |
| 4 | Texture too smooth | SMOOTHNESS WARNING | 🟡 Yellow |
| 5 | All checks pass | SHIELD: VERIFIED REAL | 🟢 Green |

## Weight Loading Strategy

The `ffpp_c23.pth` weights use a specific format:
- **Key prefix:** `model.` is stripped (maps to `timm` inner model)
- **FC remapping:** `last_linear.1.weight/bias` → `fc.weight/bias`
- **Strict loading:** `strict=True` in export pipeline (276/276 keys)

## Key Constants (config.yaml)

| Parameter               | Value                         |
|-------------------------|-------------------------------|
| Input Resolution        | 299 × 299 px                  |
| Normalization Mean      | [0.5, 0.5, 0.5]              |
| Normalization Std       | [0.5, 0.5, 0.5]              |
| Face Detection Model    | MediaPipe FaceLandmarker      |
| Detection Confidence    | 0.5                           |
| Confidence Threshold    | 0.89 (89%)                    |
| Blink (EAR) Threshold   | 0.21                          |
| Blink Time Window       | 10 seconds                    |
| Laplacian Threshold     | 50                            |
| Escape Key              | ESC (keycode 27)              |

## Development Roadmap

| Level | Task                              | Status       |
|-------|-----------------------------------|--------------|
| 1.0   | Core XceptionNet + webcam loop    | ✅ Complete   |
| 1.5   | FPS optimization (ONNX migration) | ✅ Complete   |
| 2.0   | ONNX export for AMD Ryzen AI NPU  | ✅ Complete   |
| 2.5   | V2 ONNX Runtime engine            | ✅ Complete   |
| 3.0   | INT8 static quantization (QDQ)    | ✅ Complete   |
| 3.5   | INT8 deployment engine            | ✅ Complete   |
| 4.0   | AMD Ryzen AI NPU live deploy      | 📋 Pending (requires AMD hardware) |

## Model Variants

| Model | Size | Format | Engine File |
|---|---|---|---|
| `ffpp_c23.pth` | 79.65 MB | PyTorch | `shield_xception.py` |
| `shield_ryzen_v2.onnx` | 79.31 MB | FP32 ONNX | `v2_onnx.py` |
| `shield_ryzen_int8.onnx` | 20.49 MB | INT8 QDQ ONNX | `v3_int8_engine.py` |

## ONNX Compatibility Notes

- **Opset 17:** Exported with opset 17 for 2026 hardware parity
- **Dynamic batch axis:** Supports variable batch size
- **QDQ format:** QuantizeLinear/DequantizeLinear nodes for NPU compatibility
- **Graph optimized:** Identity and Dropout nodes pruned
- **Softmax baked in:** Output sums to 1.0 (no post-processing needed)

## File Reference

| File                     | Purpose                                          |
|--------------------------|--------------------------------------------------|
| `shield_xception.py`    | Core PyTorch engine — real-time deepfake detection|
| `export_onnx.py`        | PyTorch → ONNX export pipeline                   |
| `v2_onnx.py`            | V2 ONNX Runtime engine (FP32)                    |
| `quantize_int8.py`      | FP32 → INT8 quantization pipeline                |
| `v3_int8_engine.py`     | V3 INT8 deployment engine (Diamond Tier)          |
| `shield_utils.py`       | Shared utility functions & config loader          |
| `config.yaml`           | Tunable security parameters                       |
| `ffpp_c23.pth`          | Pre-trained Xception weights (FF++ c23)           |
| `shield_ryzen_v2.onnx`  | FP32 ONNX model                                  |
| `shield_ryzen_int8.onnx`| INT8 quantized ONNX model                        |
| `face_landmarker.task`  | MediaPipe FaceLandmarker model                    |
| `GEMINI.md`             | Agent workspace rules & guardrails                |
| `docs/architecture.md`  | This file — architecture reference                |
