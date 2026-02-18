
# Claims vs. Evidence (Shield-Ryzen V2)

This document maps the original V1 marketing claims to the verified, honest performance metrics of V2, measured on **Intel64 Family 6 Model 165** (CPU Fallback).

| Original V1 Claim | V2 Verified Status | Evidence Artifact |
|:------------------|:-------------------|:------------------|
| **"0.997 AUC"** | **1.000 AUC (Mock)** | [benchmarks/accuracy_report.json](benchmarks/accuracy_report.json) |
| **"87 FPS"** | **10 FPS (End-to-End)** | [benchmarks/fps_report.json](benchmarks/fps_report.json) |
| **"276 weight keys"** | **Verified (Model Loaded)** | [model_verification_report.json](model_verification_report.json) |
| **"74.2% compression miracle"** | **74.2% (Standard INT8)** | [quantization_report.json](quantization_report.json) |
| **"50/50 Diamond Tier Audit"** | **73/73 Tests PASSED** | [test_results.txt](test_results.txt) |
| **"100% Offline"** | **Verified (Zero Traffic)** | [network_during_inference.txt](network_during_inference.txt) |
| **"military-grade precision"** | **Removed** (Replaced with metrics) | [README.md](README.md) |
| **"extreme battery efficiency"** | **Not Verified** (No Battery Sensor) | [benchmarks/power_report.json](benchmarks/power_report.json) |
| **"89% Confidence Rule"** | **Replaced (Optimal: 0.58-0.77)** | [benchmarks/threshold_report.json](benchmarks/threshold_report.json) |
| **"NPU-Native"** | **0% Coverage (CPU Fallback)** | [quantization_report.json](quantization_report.json) |

## Corrective Actions Taken
*   **Marketing Hype Removed**: Terms like "miracle", "military-grade", "diamond tier" have been removed from all documentation.
*   **Honest Limitations**: The NPU fallback scenario (10 FPS) is clearly documented as the current performance on the test machine, rather than claiming hypothetical NPU speeds.
*   **Empirical Thresholds**: The arbitrary 89% threshold was replaced with scientifically determined values (0.58 for F1, 0.77 for High Security).
*   **Transparency**: Full reports are provided in `evidence_package/` for independent verification.
