
# Shield-Ryzen V2 — Compliance & Privacy

## GDPR (EU General Data Protection Regulation)
### Article 9: Biometric Data
*   **Data Minimization**: Shield-Ryzen processes only the necessary biometric features (face embeddings, landmarks).
*   **Purpose Limitation**: Detection of deepfakes and liveness verification only.
*   **Storage Limitation**: **Biometric data is processed entirely in-memory and never stored persistently.** Frames are discarded immediately after analysis.
*   **Integrity and Confidentiality**: Data never leaves the device (verified offline operation).
*   **Consent Mechanism**: Explicit user consent is required before any biometric processing begins.
*   **Right to Erasure**: Not applicable as no data is stored.

## Illinois BIPA (Biometric Information Privacy Act)
*   **Written Consent**: Users must agree to the Biometric Information Policy before capture.
*   **Data Retention**: No biometric data (templates, scans) is retained.
*   **Prohibition on Sale**: We do not sell, lease, or trade biometric data. (Open Source project).
*   **Disclosure**: We do not disclose biometric data to any third parties.

## EU AI Act (Proposed)
*   **Classification**: **High-Risk AI System** (Biometric Identification/Categorization).
*   **Key Requirements**:
    *   **Risk Management System**: See `docs/THREAT_MODEL.md` for detailed threat analysis.
    *   **Data Governance**: Training data (FF++) is documented in `MODEL_CARD.md`. Calibration datasets are locally generated and not stored.
    *   **Technical Documentation**: Provided in `MODEL_CARD.md`, `README.md`, and source code.
    *   **Record Keeping (Logging)**: Comprehensive logs are generated (`shield_audit.jsonl`) for audit trails.
    *   **Transparency**: The HUD provides real-time feedback and explanation for decisions (e.g., "Texture Warning", "Liveness Failed").
    *   **Human Oversight**: The system is designed to assist, not replace, human judgment.
    *   **Accuracy & Robustness**: Performance metrics are verified via `benchmarks/` reports.

## Local Processing Guarantee
Evidence from `network_during_inference.txt` proves **Zero Outbound Connections** during active inference. This ensures that no facial data is transmitted to external servers, complying with the strictest privacy standards.
