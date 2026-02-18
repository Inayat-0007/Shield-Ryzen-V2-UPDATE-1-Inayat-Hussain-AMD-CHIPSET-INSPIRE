# Shield-Ryzen Model Card

## Model Details
- **Name:** Shield-Xception (v2)
- **Version:** 2.0 (Verified Integrity)
- **Date:** February 2026
- **Developer:** Inayat Hussain | AMD Slingshot
- **Architecture:** XceptionNet (`legacy_xception` variant from `timm`)
- **License:** Proprietary / Competition Use

## Intended Use
- **Primary Use Case:** Real-time deepfake detection for video streams.
- **Hardware Target:** NVIDIA RTX 3050 (Dev), AMD Ryzen AI NPU (Prod).
- **Input:** Face crops (299x299 RGB).
- **Output:** Binary classification (Real vs Fake).

## Architecture & Logic
- **Core Engine:** Deep Convolutional Neural Network (Xception).
- **Key Mechanism:** Depthwise Separable Convolutions.
  > *Note:* Unlike standard convolutions, these learned filters efficiently detect microscopic digital artifacts and texture inconsistencies common in GAN/Diffusion generated faces.
- **Input Normalization:**
  - Mean: `[0.5, 0.5, 0.5]`
  - Std:  `[0.5, 0.5, 0.5]`
  - Range: `[-1.0, 1.0]`

## Integrity & Verification
- **Checkpoint File:** `ffpp_c23.pth`
- **File Size:** 83,519,096 bytes
- **Parameters:** 20,811,050 (approx)
- **Weight Keys:** 276 (Verified)
- **SHA-256 Hash:** `8bcb10c1567d66bca32776b4c4b8f9e037be37722270e0c65643f7a2c781d762`
- **Verification Status:** ✅ PASS (Architecture matches `timm.legacy_xception`)

## Performance & Robustness
- **Training Data:** FaceForensics++ (c23 compression).
- **Adversarial Robustness (PGD eps=0.03):**
  - Baseline Confidence: ~66% (Reference Image)
  - Under Attack: ~0%
  - *Vulnerability:* High susceptibility to gradient-based attacks (Standard for non-adversarially trained models).
  - *Mitigation:* System-level checks (Liveness, Texture, Consistency) required.

## Calibration
- **Temperature Scaling:** Enabled (T=1.50).
- **Confidence Calibration:** Softmax probabilities are rescaled to reduce overconfidence.

## Version History
- **v1.0**: Initial prototype (unverified).
- **v2.0**: Added cryptographic hash check, architectural audit, and temperature scaling.
- **v2.1**: INT8 Quantization (74% compression), NPU Integration (BlazeFace 1.3ms).

## Optimization (Part 5)
- **Quantization:** INT8 Static (QDQ) via `quantize_int8.py`.
- **Compression:** 83MB → 21MB (74.2% reduction).
- **Accuracy Agreement:** 100% on test set.
- **Detector:** MediaPipe BlazeFace (ONNX) → 1.3ms latency (vs 15ms CPU).
- **Hardware Target:** Ready for Ryzen AI NPU (VitisAI EP compatible).
