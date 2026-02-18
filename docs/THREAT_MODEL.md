
# Shield-Ryzen V2 — Threat Model & Security Posture

## What Shield-Ryzen V2 CAN detect:
*   **VirtualDeepFake**: Pre-rendered deepfake videos played via virtual camera software (obs-vcam, vcam).
*   **FF++ Methods**: Face-swap deepfakes from methods in FaceForensics++ c23, including Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
*   **Replay Attacks**: Cheap replay attacks (e.g., phone screen held in front of a laptop camera).
*   **Unnatural Textures**: Unnaturally smooth AI-generated faces (via adaptive Laplacian checks).
*   **Liveness Anomalies**: Non-blinking video streams (via EAR liveness with head-pose compensation).
*   **Temporal Artifacts**: Temporal flickering artifacts (via temporal module hysteresis).
*   **Spectral Anomalies**: GAN spectral artifacts (via FFT-based frequency module).
*   **Lip-Sync Mismatch**: Audio-visual correlation mismatches (if verified audio module is enabled).

## What Shield-Ryzen V2 CANNOT reliably detect:
*   **Unknown Deepfakes**: Deepfake methods not present in the training data (e.g., highly advanced diffusion-based models).
*   **High-Quality Liveness**: High-quality deepfakes that perfectly preserve natural blinking and micro-expressions.
*   **Adversarial Attacks**: Adversarial examples crafted specifically against this XceptionNet model (white-box attacks).
*   **3D Masks**: High-quality 3D printed masks with realistic texture and depth.
*   **OS-Level Spoofing**: Driver-level camera spoofing (injected frames below the OS capture layer).
*   **Video Compression**: Heavily compressed video call streams (may cause higher false positive rate due to artifacts).
*   **Social Engineering**: Coercion or social engineering attacks targeting the user.
*   **Hardware Attacks**: Physical sensor injection attacks or tampering with the camera hardware.

## Security Assumptions:
*   **Hardware Integrity**: The camera hardware is physically intact and has not been compromised.
*   **OS Integrity**: The operating system kernel and drivers are not compromised by an attacker.
*   **Secure Boot**: The NPU secure boot chain (AMD Secure Processor) is intact and functioning.
*   **User Consent**: The user explicitly consents to biometric processing and understands the limitations.

## What is NOT tested (Out of Scope):
*   **Adversarial Robustness**: We have not conducted extensive adversarial robustness testing (e.g., FGSM, PGD attacks).
*   **Diffusion Models**: Performance against SOTA diffusion-based video generation is unverified.
*   **Hardware Failover**: NPU failover under extreme load conditions (thermal throttling) is not fully characterized.

## Mitigation Strategy
Shield-Ryzen V2 is designed as a *layered defense*. No single check is foolproof. By combining multiple detection methods (texture, liveness, frequency, temporal consistency), we aim to raise the bar for attackers significantly, forcing them to use more sophisticated and costly methods.
