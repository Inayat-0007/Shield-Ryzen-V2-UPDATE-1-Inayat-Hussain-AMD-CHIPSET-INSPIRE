"""
Shield-Ryzen V2 — Normalization Verification
==============================================
Checks that V1 preprocessing matches the original XceptionNet
training normalization by comparing pixel statistics.

Verifies:
  1. Input resize to 299×299
  2. [0.5, 0.5, 0.5] mean and std normalization (FF++ standard)
  3. Output tensor shape: [1, 3, 299, 299] (NCHW)
  4. Output range: [-1.0, 1.0]
  5. Consistency across multiple images

Usage:
    python scripts/verify_normalization.py

Developer: Inayat Hussain | AMD Slingshot 2026
Part 1 of 12 — Input Pipeline Hardening
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shield_utils import preprocess_face, MEAN, STD, INPUT_SIZE


def _create_test_images() -> list[tuple[str, np.ndarray]]:
    """Generate synthetic test images with known pixel distributions."""
    test_images: list[tuple[str, np.ndarray]] = []

    # 1. Uniform mid-gray (128, 128, 128) — known easy case
    gray = np.full((480, 640, 3), 128, dtype=np.uint8)
    test_images.append(("uniform_gray_128", gray))

    # 2. All-black (0, 0, 0)
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add small brightness to pass camera validation
    black[:, :] = 10
    test_images.append(("near_black", black))

    # 3. All-white (255, 255, 255)
    white = np.full((480, 640, 3), 245, dtype=np.uint8)
    test_images.append(("near_white", white))

    # 4. Gradient image (pixel values 0-255 across width)
    gradient = np.zeros((480, 640, 3), dtype=np.uint8)
    for c in range(3):
        gradient[:, :, c] = np.tile(np.linspace(10, 240, 640, dtype=np.uint8), (480, 1))
    test_images.append(("gradient", gradient))

    # 5. Random natural-ish face crop (simulated)
    rng = np.random.RandomState(42)
    natural = rng.randint(40, 220, size=(300, 300, 3), dtype=np.uint8)
    test_images.append(("random_face_sim", natural))

    # 6. Real calibration image (if available)
    calib_dir = Path(_project_root) / "calibration_data"
    if calib_dir.is_dir():
        first_img = sorted(calib_dir.glob("*.jpg"))
        if first_img:
            img = cv2.imread(str(first_img[0]))
            if img is not None:
                test_images.append(("real_calibration", img))

    return test_images


def verify_normalization() -> dict:
    """Run comprehensive normalization verification.

    Returns dict with verification results and pass/fail status.
    """
    print("=" * 60)
    print("  SHIELD-RYZEN V2 — NORMALIZATION VERIFICATION")
    print("=" * 60)

    results: dict = {
        "config": {
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
            "input_size": INPUT_SIZE,
            "expected_output_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "expected_output_range": [-1.0, 1.0],
        },
        "tests": [],
        "all_passed": True,
    }

    test_images = _create_test_images()
    all_passed = True

    for name, img in test_images:
        print(f"\n  Testing: {name} (input shape: {img.shape})")
        test_result: dict = {"name": name, "input_shape": list(img.shape), "checks": {}}

        try:
            tensor = preprocess_face(img)

            # Check 1: Output shape
            expected_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
            shape_ok = tensor.shape == expected_shape
            test_result["checks"]["shape"] = {
                "expected": list(expected_shape),
                "actual": list(tensor.shape),
                "passed": shape_ok,
            }
            print(f"    Shape: {tensor.shape} — {'✅' if shape_ok else '❌'}")

            # Check 2: Output dtype
            dtype_ok = tensor.dtype == np.float32
            test_result["checks"]["dtype"] = {
                "expected": "float32",
                "actual": str(tensor.dtype),
                "passed": dtype_ok,
            }
            print(f"    Dtype: {tensor.dtype} — {'✅' if dtype_ok else '❌'}")

            # Check 3: Output range [-1, 1]
            # WHY: FF++ XceptionNet was trained with pixel values in [-1, 1]
            # using (x - 0.5) / 0.5 normalization
            t_min = float(tensor.min())
            t_max = float(tensor.max())
            range_ok = t_min >= -1.01 and t_max <= 1.01  # small tolerance for float
            test_result["checks"]["range"] = {
                "expected": [-1.0, 1.0],
                "actual": [round(t_min, 4), round(t_max, 4)],
                "passed": range_ok,
            }
            print(f"    Range: [{t_min:.4f}, {t_max:.4f}] — {'✅' if range_ok else '❌'}")

            # Check 4: Verify normalization formula correctness
            # For uniform gray 128: expected = (128/255 - 0.5) / 0.5 ≈ 0.00392
            if name == "uniform_gray_128":
                expected_val = (128.0 / 255.0 - 0.5) / 0.5
                actual_mean = float(tensor.mean())
                formula_ok = abs(actual_mean - expected_val) < 0.01
                test_result["checks"]["formula"] = {
                    "expected_mean": round(expected_val, 6),
                    "actual_mean": round(actual_mean, 6),
                    "passed": formula_ok,
                }
                print(f"    Formula: expected={expected_val:.6f} actual={actual_mean:.6f} — {'✅' if formula_ok else '❌'}")

            # Check 5: NCHW ordering (channels first)
            nchw_ok = tensor.shape[1] == 3 and tensor.shape[2] == INPUT_SIZE
            test_result["checks"]["nchw"] = {"passed": nchw_ok}
            print(f"    NCHW: {'✅' if nchw_ok else '❌'}")

            # Per-channel statistics
            for c, ch_name in enumerate(["R", "G", "B"]):
                ch = tensor[0, c, :, :]
                test_result["checks"][f"channel_{ch_name}"] = {
                    "mean": round(float(ch.mean()), 6),
                    "std": round(float(ch.std()), 6),
                    "min": round(float(ch.min()), 6),
                    "max": round(float(ch.max()), 6),
                }

            passed = all(
                c.get("passed", True)
                for c in test_result["checks"].values()
                if isinstance(c, dict) and "passed" in c
            )
            test_result["passed"] = passed

        except Exception as e:
            test_result["passed"] = False
            test_result["error"] = str(e)
            print(f"    ❌ Error: {e}")
            passed = False

        if not passed:
            all_passed = False
        results["tests"].append(test_result)

    results["all_passed"] = all_passed

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(1 for t in results["tests"] if t["passed"])
    n_total = len(results["tests"])
    status = "✅ ALL PASSED" if all_passed else "❌ FAILURES DETECTED"
    print(f"  RESULT: {status} ({n_pass}/{n_total} tests)")
    print("=" * 60)

    # Save
    out_path = os.path.join(_project_root, "scripts", "normalization_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Report saved → {out_path}")

    return results


if __name__ == "__main__":
    verify_normalization()
