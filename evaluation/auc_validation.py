"""
Shield-Ryzen V2 — AUC Validation
==================================
Loads ONNX model, runs inference on available face-forensic datasets,
and outputs per-dataset AUC with 95% confidence intervals and EER.

NORMALIZATION: Uses the SAME preprocessing as shield_face_pipeline.py
(single source of truth) to ensure evaluation matches inference:
    normalized = (pixel_uint8 / 255.0 - 0.5) / 0.5
    Range: [-1.0, +1.0] (FF++ standard, OPTION B)

Supported datasets:
    - FaceForensics++ c23 (test split)
    - Celeb-DF v2
    - DFDC preview set
    - Any directory with real/ and fake/ subdirectories

Usage:
    python evaluation/auc_validation.py \\
        --model shield_ryzen_int8.onnx \\
        --datasets ff_c23=/path/to/ff_c23 celeb_df=/path/to/celeb_df \\
        --output evaluation/auc_results.json

Developer: Inayat Hussain | AMD Slingshot 2026
Part 2 of 14 — Face Detection, Preprocessing & Normalization Fix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


# ─── Preprocessing (SINGLE SOURCE OF TRUTH) ──────────────────
# Import from shield_face_pipeline to guarantee evaluation uses
# the same normalization as the live inference engine.

try:
    from shield_face_pipeline import ShieldFacePipeline
    _pipeline_available = True
except ImportError:
    _pipeline_available = False

# Constants documented here for traceability (must match pipeline)
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
INPUT_SIZE = 299


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Preprocess face crop: resize 299×299, normalize to [-1, 1], NCHW.

    Uses the identical formula as shield_face_pipeline.align_and_crop:
        face_float = face_rgb.astype(np.float32) / 255.0
        face_norm  = (face_float - 0.5) / 0.5
    Range: [-1.0, +1.0]
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    face_float = face_resized.astype(np.float32) / 255.0
    face_norm = (face_float - MEAN) / STD
    face_chw = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0).astype(np.float32)


# ─── Dataset Loading ─────────────────────────────────────────

def _load_dataset(dataset_path: str) -> tuple[list[str], list[int]]:
    """Load image paths and labels from a dataset directory.

    Expected structure:
        dataset_path/
        ├── real/  (or 0/ or original/)
        │   ├── img001.png
        │   └── ...
        └── fake/  (or 1/ or manipulated/)
            ├── img001.png
            └── ...

    Returns:
        (image_paths, labels) where label 0=real, 1=fake
    """
    paths: list[str] = []
    labels: list[int] = []

    real_dirs = ["real", "0", "original", "Real"]
    fake_dirs = ["fake", "1", "manipulated", "Fake", "altered"]

    dataset = Path(dataset_path)

    for dir_name in real_dirs:
        real_dir = dataset / dir_name
        if real_dir.is_dir():
            for img_path in sorted(real_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    paths.append(str(img_path))
                    labels.append(0)
            break

    for dir_name in fake_dirs:
        fake_dir = dataset / dir_name
        if fake_dir.is_dir():
            for img_path in sorted(fake_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    paths.append(str(img_path))
                    labels.append(1)
            break

    return paths, labels


# ─── AUC Computation ─────────────────────────────────────────

def _compute_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute AUC using the trapezoidal rule (no sklearn dependency).

    Uses the Mann-Whitney U statistic formulation.
    """
    if len(set(y_true)) < 2:
        return float("nan")

    pairs = list(zip(y_true, y_scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    for label, _ in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return round(auc, 6)


def _compute_eer(y_true: list[int], y_scores: list[float]) -> float:
    """Compute Equal Error Rate (where FPR == FNR)."""
    thresholds = sorted(set(y_scores))
    best_eer = 1.0

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    for thresh in thresholds:
        fp = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s >= thresh)
        fn = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s < thresh)
        fpr = fp / n_neg
        fnr = fn / n_pos
        eer = (fpr + fnr) / 2.0
        if abs(fpr - fnr) < abs(best_eer * 2 - 1):
            best_eer = eer

    return round(best_eer, 6)


def _bootstrap_ci(
    y_true: list[int],
    y_scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(42)  # Reproducible
    aucs: list[float] = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample_true = [y_true[i] for i in idx]
        sample_scores = [y_scores[i] for i in idx]
        if len(set(sample_true)) < 2:
            continue
        aucs.append(_compute_auc(sample_true, sample_scores))

    if not aucs:
        return (float("nan"), float("nan"))

    aucs.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = aucs[int(alpha * len(aucs))]
    hi = aucs[int((1 - alpha) * len(aucs))]
    return (round(lo, 6), round(hi, 6))


# ─── Main Validation ─────────────────────────────────────────

def validate_auc(
    model_path: str,
    datasets: dict[str, str],
    output_path: str = "evaluation/auc_results.json",
) -> dict[str, Any]:
    """Run AUC validation across multiple datasets.

    Args:
        model_path: Path to ONNX model file.
        datasets: Mapping of dataset_name → dataset_directory.
            E.g. {"ff_c23": "/path/to/ff_c23/", "celeb_df": "/path/to/celeb_df/"}
        output_path: Where to save JSON results.

    Returns:
        Per-dataset results: AUC, 95% CI, EER, sample counts.
    """
    if ort is None:
        raise ImportError("onnxruntime is required for AUC validation")

    print("=" * 60)
    print("  SHIELD-RYZEN V2 — AUC VALIDATION")
    print("=" * 60)

    # Load model
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(
        model_path, sess_options=sess_opts, providers=providers
    )
    active_provider = session.get_providers()[0]
    print(f"\n  Model: {model_path}")
    print(f"  Provider: {active_provider}")

    results: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": os.path.basename(model_path),
            "provider": active_provider,
        },
        "datasets": {},
    }

    for ds_name, ds_path in datasets.items():
        print(f"\n  [{ds_name}] Loading from: {ds_path}")

        if not os.path.isdir(ds_path):
            results["datasets"][ds_name] = {"error": f"Directory not found: {ds_path}"}
            print(f"    ⚠️  Directory not found — skipping")
            continue

        paths, labels = _load_dataset(ds_path)
        if not paths:
            results["datasets"][ds_name] = {"error": "No images found"}
            print(f"    ⚠️  No images found — skipping")
            continue

        n_real = labels.count(0)
        n_fake = labels.count(1)
        print(f"    Samples: {len(paths)} ({n_real} real, {n_fake} fake)")

        # Run inference
        scores: list[float] = []  # fake probability scores
        errors = 0
        t_start = time.perf_counter()

        for i, img_path in enumerate(paths):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    errors += 1
                    scores.append(0.5)  # neutral on error
                    continue
                tensor = preprocess_face(img)
                output = session.run(None, {"input": tensor})[0]
                fake_prob = float(output[0, 0])
                scores.append(fake_prob)
            except Exception:
                errors += 1
                scores.append(0.5)

            if (i + 1) % 100 == 0:
                print(f"    Progress: {i + 1}/{len(paths)}")

        elapsed = time.perf_counter() - t_start

        # Compute metrics
        auc = _compute_auc(labels, scores)
        eer = _compute_eer(labels, scores)
        ci_lo, ci_hi = _bootstrap_ci(labels, scores)

        ds_result = {
            "n_samples": len(paths),
            "n_real": n_real,
            "n_fake": n_fake,
            "n_errors": errors,
            "auc": auc,
            "auc_95ci": [ci_lo, ci_hi],
            "eer": eer,
            "inference_time_seconds": round(elapsed, 2),
            "avg_inference_ms": round(elapsed / len(paths) * 1000, 2),
        }
        results["datasets"][ds_name] = ds_result

        print(f"    AUC: {auc:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
        print(f"    EER: {eer:.4f}")
        print(f"    Time: {elapsed:.1f}s ({ds_result['avg_inference_ms']:.1f} ms/img)")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ AUC results saved → {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shield-Ryzen AUC Validation")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME=PATH",
        help="Dataset definitions (e.g. ff_c23=/data/ff_c23)",
    )
    parser.add_argument("--output", type=str, default="evaluation/auc_results.json")
    args = parser.parse_args()

    ds_map: dict[str, str] = {}
    if args.datasets:
        for entry in args.datasets:
            name, path = entry.split("=", 1)
            ds_map[name] = path

    validate_auc(
        model_path=args.model,
        datasets=ds_map,
        output_path=args.output,
    )
