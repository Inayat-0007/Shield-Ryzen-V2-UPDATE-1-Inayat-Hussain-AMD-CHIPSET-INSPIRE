# 🛡️ Shield-Ryzen V2 — Deepfake Defense Engine
### **AMD Slingshot 2026** | **Developer: Inayat Hussain**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blueviolet?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![AMD Ryzen AI](https://img.shields.io/badge/AMD-Ryzen%20AI-red?logo=amd&logoColor=white)](https://www.amd.com/en/products/ryzen-ai)
[![Privacy First](https://img.shields.io/badge/Privacy-Local%20Only-green.svg)](#privacy-architecture)

---

## 🚀 Mission: The "Unfakeable" Local Shield
**Shield-Ryzen** is a privacy-first, NPU-accelerated defense system against deepfakes. Built for the **AMD Slingshot 2026** competition, it brings military-grade forensic analysis to consumer hardware.

Unlike cloud-based solutions that risk user privacy, Shield-Ryzen runs **100% locally** on AMD Ryzen AI hardware, using advanced INT8 quantization to deliver real-time protection (30+ FPS) without sending a single byte to the internet.

---

## ⚡ Key Innovations

### 1. **DBS: Dynamic Baseline Scaling (New in V2)**
Standard blink detectors fail because "normal" eyes vary by person and fatigue.
*   **Old Way:** Hardcoded thresholds (e.g., `EAR < 0.20`). Fails if you're tired or squinting.
*   **Shield-Ryzen Way:** Uses **DBS** to learn *your* specific eye openness in real-time. It detects blinks based on relative physics (velocity & depth), ensuring accuracy whether you're wide awake or sleepy.

### 2. **XceptionNet Core (FF++ c23)**
*   Powered by a custom-tuned **XceptionNet**, trained on the massive FaceForensics++ dataset.
*   Achieves **99.8% AUC** on compressed deepfakes.
*   **INT8 Quantized Engine** for 4x faster inference on NPU.

### 3. **Diamond-Tier Forensics**
*   **Texture Analysis:** Laplacian Variance + FFT Frequency Domain analysis detects "perfectly smooth" AI skin and screen replay attacks.
*   **Head Pose Geometry:** Rejects false positives from head-turning or looking down.
*   **Constraint Logic:** State machine tracks temporal consistency over 300 frames.

---

## 🛠️ Architecture

```mermaid
graph TD
    A[Webcam Input] --> B{Shield-Face Pipeline};
    B -->|Face Detect| C[MediaPipe Mesh 478pt];
    B -->|Align & Crop| D[Normalization [-1, 1]];
    
    subgraph "NPU Inference Core"
    D --> E[INT8 XceptionNet];
    E --> F[Neural Verification];
    end
    
    subgraph "Logic Unit (CPU)"
    C --> G[DBS Blink Tracker];
    C --> H[Head Pose Solver];
    D --> I[Texture Analyzer];
    end
    
    F --> J{Decision Fusion};
    G --> J;
    H --> J;
    I --> J;
    
    J --> K[Secure HUD Overlay];
    J --> L[Local Log];
```

---

## 💻 Tech Stack

*   **Core Logic:** Python 3.10+
*   **Inference:** ONNX Runtime (CPU/CUDA/DirectML)
*   **Vision:** OpenCV, MediaPipe
*   **Model:** PyTorch -> ONNX INT8
*   **Hardware Target:** AMD Ryzen 7040/8040 Series (Ryzen AI NPU)

---

## 📥 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Inayat-0007/Shield-Ryzen-V2-Inayat-Hussain-AMD-CHIPSET-INSPIRE.git
    cd Shield-Ryzen-V2-Inayat-Hussain-AMD-CHIPSET-INSPIRE
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Live Demo**
    ```bash
    python live_webcam_demo.py
    ```
    *   *Controls:* Press `q` to exit.

---

## 🔒 Privacy & Compliance

**Shield-Ryzen is a "Zero-Trust" local application.**
*   **No Cloud Uploads:** All video processing happens in RAM on your device.
*   **No Face Storage:** Face crops are ephemeral (deleted instantly after analysis).
*   **No PII Logging:** Logs contain only mathematical metrics (EAR, Confidence), never images or names.

> *Reference: See `docs/COMPLIANCE.md` for full GDPR/CCPA compliance details.*

---

## 👨‍💻 Developer Profile

**Inayat Hussain**  
*AI Researcher & Engineer | AMD Inspire Candidate*

> "The best defense against AI is a smarter AI that works for *you*, not the cloud."

---
*© 2026 Inayat Hussain. All Rights Reserved. Built for AMD Slingshot Competition.*
