# Shield-Ryzen V2: Real-Time Deepfake Detection System
**Target Hardware:** AMD Ryzen™ AI NPU (via Vitis AI EP)
**Developer:** Inayat Hussain | AMD Slingshot 2026

![Status](https://img.shields.io/badge/Status-Active-green)
![NPU](https://img.shields.io/badge/Target-Ryzen2-orange)
![Python](https://img.shields.io/badge/Python-3.13-blue)

## 🛡️ Project Overview
Shield-Ryzen V2 is an enterprise-grade deepfake detection engine specifically optimized for AMD Ryzen AI mobile processors. It combines neural analysis (XceptionNet) with biometric verification (Blink, Heartbeat) and forensic signal processing to detect sophisticated AI-generated content in real-time (<30ms per frame).

### Key Features
- **Neural Engine**: Quantized INT8 XceptionNet running on NPU.
- **Biometric Liveness**: Detects blink patterns and rPPG heartbeat signals.
- **Signal Forensics**: Analyzes frequency domain (FFT) and codec artifacts (8x8 blocking).
- **Enterprise Security**: Encrypted face re-identification and immutable audit trails.
- **Accessibility**: WCAG 2.1 AA compliant HUD with audio alerts.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- AMD Ryzen 7040/8040 Series (recommended for NPU acceleration)
- Webcam

### 2. Installation
```bash
pip install -r requirements.txt
# Optional: Install Vitis AI driver for NPU
```

### 3. Running the System
Launch the optimized engine:
```bash
python start_shield.py
```
Options:
- `--source <id>`: Use specific camera (default 0) or video file.
- `--model <path>`: Load custom ONNX model.
- `--audit`: Enable secure logging.
- `--cpu`: Force CPU execution (disable NPU).

## 📂 Project Structure
```
Shield-Ryzen-V2/
├── shield_engine.py       # Core orchestrator (Triple-Buffer loop)
├── shield_xception.py     # Neural network wrapper (ONNX/PyTorch)
├── start_shield.py        # Main launcher script
├── v3_xdna_engine.py      # AMD NPU optimization subclass
├── quantize_ryzen.py      # INT8 Quantization Script
├── plugins/               # Modular security layers
│   ├── blink_detector.py
│   ├── frequency_analyzer.py
│   ├── codec_forensics.py
│   └── arcface_reid.py
├── security/              # Enterprise compliance
│   └── audit_trail.py
├── docs/                  # Documentation
│   ├── architecture.md
│   ├── THREAT_MODEL.md
│   └── COMPLIANCE.md
└── tests/                 # Validation suite
```

## 🛠️ Validation
Run the comprehensive test suite:
```bash
python validate_system.py
```
This checks all subsystems including NPU loading, plugin logic, and security chains.

## ⚖️ Compliance
Shield-Ryzen V2 is designed in adherence to:
- **GDPR Article 9**: Explicit consent & data minimization.
- **Illinois BIPA**: Secure retention policies.
- **EU AI Act**: Transparency & Auditability.

See `docs/COMPLIANCE.md` for details.

## License
MIT License.
Copyright (c) 2026 Inayat Hussain.
