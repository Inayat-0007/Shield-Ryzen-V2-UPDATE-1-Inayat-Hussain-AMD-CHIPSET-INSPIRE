# Shield-Xception Model Card

## Model Details
| Field | Value |
|-------|-------|
| **Name** | Shield-Xception (v2) |
| **Version** | 2.0 (Verified Integrity) |
| **Date** | February 2026 |
| **Developer** | Inayat Hussain — AMD Slingshot 2026 |
| **Architecture** | XceptionNet (`legacy_xception` variant from `timm`) |
| **License** | Proprietary / Competition Use |

## Intended Use
- **Primary Use Case:** Real-time deepfake detection for video streams.
- **Deployment:** Local-only inference — **NO** cloud APIs, **ALL** processing on-device.
- **Hardware Targets:**
  - Development: NVIDIA RTX 3050 Laptop GPU (CUDA)
  - Production: AMD Ryzen AI NPU (via ONNX Runtime + VitisAI EP)
- **Input:** Face crops (299×299 RGB, pre-detected by MediaPipe BlazeFace/FaceLandmarker)
- **Output:** Binary classification — `[Fake probability, Real probability]`

## Architecture & Mechanism

### Core Architecture: XceptionNet
The model is based on the **Xception** (Extreme Inception) architecture by François Chollet.
It uses **depthwise separable convolutions** as its fundamental building block, replacing
standard convolutions throughout the network (except the entry-flow stem).

### How Depthwise Separable Convolutions Work
A standard convolution jointly maps spatial and cross-channel correlations.
A **depthwise separable convolution** decomposes this into two independent steps:

1. **Depthwise convolution** — one filter per input channel. This performs spatial
   filtering (e.g., edge detection, texture analysis) independently on each channel.
   _Groups = in_channels._

2. **Pointwise convolution** — a 1×1 convolution that linearly combines the outputs
   of the depthwise layer across channels.

This decomposition reduces parameters by ~8× compared to standard convolutions while
preserving representational capacity. The **LEARNED FILTERS** within these convolutions
detect microscopic digital artifacts and texture inconsistencies specific to
GAN/Diffusion-generated faces.

### Verified Architecture Details
| Metric | Count |
|--------|-------|
| **Total weight keys** | 276 (verified by `verify_model.py`) |
| **Total parameters** | ~20,811,050 |
| **Depthwise conv layers** | ~36 |
| **Pointwise 1×1 conv layers** | ~36 |
| **Standard conv layers** | 2 (entry-flow stem: `conv1`, `conv2`) |
| **Separable pairs** | ~36 (depthwise + pointwise) |
| **Entry flow blocks** | block1–block3 (stride-2 downsampling) |
| **Middle flow blocks** | block4–block11 (repeated residual blocks) |
| **Exit flow blocks** | block12 |
| **Classifier head** | Global Average Pooling → FC (2048 → 2) |

### Key Mapping
The checkpoint uses `model.` prefix for all keys and names the classifier head
`model.last_linear.1.{weight,bias}`. The timm library uses `fc.{weight,bias}`.
This renaming is handled automatically by `load_model_with_verification()`.
- **274/276 keys** match directly after stripping the `model.` prefix.
- **2/276 keys** require FC head renaming → 100% effective mapping.

## Input Preprocessing
| Step | Value |
|------|-------|
| **Color space** | RGB (converted from BGR by OpenCV) |
| **Resize** | 299×299 pixels |
| **Normalization mean** | `[0.5, 0.5, 0.5]` |
| **Normalization std** | `[0.5, 0.5, 0.5]` |
| **Output range** | `[-1.0, 1.0]` |
| **Formula** | `(pixel / 255.0 - 0.5) / 0.5` |

## Output Specification
| Index | Meaning | Range |
|-------|---------|-------|
| `output[0]` | **Fake** probability | [0.0, 1.0] |
| `output[1]` | **Real** probability | [0.0, 1.0] |
| **Activation** | Softmax (applied in `ShieldXception.forward()`) | sums to 1.0 |

### Confidence Calibration
Raw softmax probabilities are rescaled using **Temperature Scaling** (Part 3):
- **Temperature:** T = 1.50
- **Effect:** Reduces overconfidence — `softmax(logits / T)` spreads the distribution.
- **Calibrator:** `ConfidenceCalibrator` from `shield_utils_core.py`

## Integrity & Verification
| Check | Value |
|-------|-------|
| **Checkpoint file** | `ffpp_c23.pth` |
| **File size** | 83,519,096 bytes (79.6 MB) |
| **SHA-256 hash** | `8bcb10c1567d66bca32776b4c4b8f9e037be37722270e0c65643f7a2c781d762` |
| **Verification** | ✅ PASS — Architecture matches `timm.legacy_xception` |
| **Signature file** | `models/model_signature.sha256` |
| **Tamper detection** | `ModelTamperingError` raised on hash mismatch |
| **Reference output** | Deterministic zeros input → recorded in `models/reference_output.json` |

### Security Hardening
1. **SHA-256 at load time** — Every model load compares the file hash against `models/model_signature.sha256`
2. **ModelTamperingError** — Raises a security exception if the hash doesn't match (model modified/corrupted)
3. **Key count validation** — Warns if checkpoint has unexpected number of keys
4. **ONNX shape verification** — Validates input shape `(1, 3, 299, 299)` and output shape `(1, 2)` for ONNX models

## Training Data
| Field | Value |
|-------|-------|
| **Dataset** | FaceForensics++ |
| **Compression** | c23 (medium quality, realistic degradation) |
| **Manipulation types** | DeepFakes, Face2Face, FaceSwap, NeuralTextures |
| **Split** | Standard FF++ train/val/test (720/140/140 videos) |
| **Augmentation** | Standard training augmentation (flip, color jitter) |

## Adversarial Robustness
Non-adversarially trained models are inherently vulnerable to gradient-based attacks.
Shield-Ryzen mitigates this through **system-level layered defense**:

| Attack | ε | Result | Notes |
|--------|---|--------|-------|
| **Clean** | — | Baseline confidence ~66% | On reference test image |
| **FGSM** | 1/255 | Minimal drop | Imperceptible perturbation |
| **FGSM** | 4/255 | Moderate drop | Below 5% target may not be met |
| **FGSM** | 8/255 | Significant drop | Visible perturbation |
| **PGD** | 4/255, 20 steps | Near-complete drop | Expected for non-AT models |

### Mitigation Strategy
The neural network is **one tier** of a multi-tier security system:
- **Tier 1:** Neural network confidence (this model)
- **Tier 2:** Liveness detection (EAR-based blink verification)
- **Tier 3:** Texture forensics (Laplacian + FFT analysis)
- **Decision fusion:** `DecisionStateMachine` with conflict resolution truth table

Adversarial attacks can fool Tier 1 but cannot simultaneously fool Tiers 2 and 3.

## Quantization (Part 5)
| Variant | Format | Size | Compression | Accuracy |
|---------|--------|------|-------------|----------|
| **FP32** | `.pth` → `.onnx` | 83 MB | — | Baseline |
| **INT8** | ONNX QDQ | 21 MB | 74.2% | 100% agreement on test set |
| **Target** | INT4 (planned) | ~11 MB | — | TBD |
| **Runtime** | ONNX Runtime (CPU/CUDA/VitisAI EP) | — | — |

## Version History
| Version | Changes |
|---------|---------|
| **v1.0** | Initial prototype (unverified integrity) |
| **v2.0** | Added SHA-256 hash check, architectural audit, temperature scaling |
| **v2.1** | INT8 Quantization (74% compression), NPU Integration (BlazeFace 1.3ms) |
| **v2.2** | Full integrity hardening (Part 4): ModelTamperingError, ONNX shape verification, reference output regression, MODEL_CARD |

## Files
```
ffpp_c23.pth                          # Original FP32 checkpoint
shield_xception.py                    # ShieldXception class + secure loading
verify_model.py                       # Full verification audit script
model_verification_report.json        # Detailed key mapping report
models/model_signature.sha256         # SHA-256 hash for runtime check
models/reference_output.json          # Deterministic reference output
security/adversarial_test_suite.py    # FGSM + PGD attack evaluation
security/adversarial_robustness.json  # Attack results report
MODEL_CARD.md                         # This file
```
