# Shield-Ryzen V2 Threat Model

## Overview
This document outlines the security capabilities and limitations of the Shield-Ryzen V2 system. It serves as a transparent disclosure of what attacks are mitigated and which remain out of scope or require additional layers.

## What We CAN Detect (with Evidence)
The system employs a multi-layered defense strategy (Neural, Biometric, Forensic) to identify:

1.  **Pre-rendered Deepfake Videos**: 
    - Detected via **Blink Analysis** (unnatural patterns) and **Heartbeat** (rPPG signal absence).
    - **Codec Forensics** identifies re-compression artifacts (macroblocking) from virtual cameras.

2.  **Face-Swap Deepfakes (e.g., DeepFaceLab, FaceSwap)**:
    - **XceptionNet** (Core Engine) trained on FaceForensics++ (c23) detects manipulation boundaries.
    - **Texture Analysis** flags unnaturally smooth skin or loss of pore detail.

3.  **Cheap Replay Attacks (Phone/Screen)**:
    - **Stereo Depth** (if dual camera available) detects flat surfaces.
    - **Skin Reflectance** plugin identifies excessive specular reflection typical of screens.
    - **Frequency Analysis** detects Moiré patterns.

4.  **Generative AI Faces (GAN/Diffusion)**:
    - **Frequency Analyzer** detects suppressed high-frequency energy (blur/smoothing artifacts) characteristic of GAN upsampling.

5.  **Adversarial Patches**:
    - **Adversarial Detector** identifies high-gradient anomalies (e.g., adversarial glasses/stickers) designed to fool neural networks.

6.  **Injection Attacks (OBS Virtual Camera)**:
    - **Codec Forensics** flags 8x8 block discrepancies and lack of sensor noise.

## What We CANNOT Reliably Detect
Attack vectors that may bypass current defenses:

1.  **Diffusion-Based Video Generation (Sora/Runway)**:
    - High-fidelity temporal consistency may fool simple blink detectors.
    - Requires continuous retraining on new generation architectures.

2.  **3D Silicon Masks (High Quality)**:
    - If mask has realistic texture and warmth (defeats texture/reflectance checks).
    - Require thermal camera or active IR for reliable detection.

3.  **Driver-Level Injection**:
    - Attacks occurring *below* the OS camera driver (e.g., kernel-level rootkits feeding raw buffers) may bypass codec checks if they simulate sensor noise perfectly.

4.  **Audio-Visual Deepfakes (Perfect Lip-Sync)**:
    - While we have a Lip-Sync verifier, advanced models (Wav2Lip) can generate convincing synchronization.

## Assumptions & Trust Boundary
1.  **Hardware Integrity**: We assume the physical camera and AMD Ryzen NPU are not compromised.
2.  **OS Security**: The host OS (Windows) is assumed to be free of kernel-level rootkits.
3.  **Local Execution**: All processing occurs locally. We trust the local memory (protected by ASLR/DEP, though Python is memory-managed).
4.  **User Consent**: The system operates under the assumption that the authorized user has consented to biometric verification.

## Data Privacy
- **No Cloud Transmission**: All biometric data is processed on-device.
- **Encryption**: Enrollment data is AES-256 encrypted at rest (`shield_crypto.py`).
- **Audit**: All decisions are logged in a tamper-evident hash chain (`security/audit_trail.py`).
