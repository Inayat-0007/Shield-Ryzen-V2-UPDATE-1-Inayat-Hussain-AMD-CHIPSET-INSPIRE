# Shield-Ryzen V2 Compliance & Regulations

## GDPR (General Data Protection Regulation) - EU
### Correct Processing of Special Category Data (Article 9)
Biometric data used for uniquely identifying a natural person is **prohibited** (Art 9(1)) unless explicit consent is given (Art 9(2)(a)).

- **Implementation**: The system requires explicit user opt-in (via `enable_challenge_response` configuration) and enrollment.
- **Data Minimization**: Biometric embeddings (128-d vectors) are stored, not raw face images (except transiently in RAM).
- **Storage Limitation**: All data is encrypted locally (`AES-256`) and never transmitted off-device. No cloud storage.
- **Right to Erasure**: Users can invoke the `secure_wipe()` utility to permanently delete all enrolled data.

## Illinois BIPA (Biometric Information Privacy Act) - US
Regulates the collection, use, and retention of biometric identifiers.

### Requirements & Compliance:
1.  **Written Policy**: A publicly available retention schedule and destruction guidelines are required.
    - **Policy**: [See THREAT_MODEL.md] We retain data only for the active session duration unless enrolled for re-identification.
2.  **Written Release**: Must obtain a written release from the subject prior to collection.
    - **Action**: UI prompts for consent upon first launch.
3.  **Prohibition on Profit**: We do not sell, lease, or trade biometric data.
4.  **Standard of Care**: Data is stored securely (encrypted) using industry-standard cryptography (`cffi` / `cryptography` library).

## EU AI Act (2024)
Classifies AI systems based on risk. Biometric identification systems are **High-Risk**.

### Requirements for High-Risk AI:
1.  **Risk Management System**: We maintain a continuous risk assessment (`THREAT_MODEL.md`).
2.  **Data Governance**: Training data (FaceForensics++) is curated to minimize bias. Calibration (`shield_utils_core.py`) adjusts for lighting/skin tone variance.
3.  **Technical Documentation**: Detailed architecture (`docs/architecture.md`) and logging (`shield_logger.py`).
4.  **Record Keeping**: An immutable audit trail (`security/audit_trail.py`) logs all system decisions for post-market monitoring.
5.  **Transparency & User Info**: The HUD (`shield_hud.py`) provides clear, accessible feedback (WCAG 2.1 AA) on system status and confidence.
6.  **Human Oversight**: The dashboard allows manual overrides and explains decisions (GradeCAM compatibility).
7.  **Accuracy & Cybersecurity**: The system is robust against adversarial attacks (`shield_adversarial.py`) and protects against data breaches (`shield_crypto.py`).

## Disclaimer
This software is provided "as is". Deployment in specific jurisdictions (e.g., public surveillance) may require additional legal review. The developers explicitly disclaim liability for misuse.
