"""
Shield-Ryzen V2 — Integrity Verification Script
==============================================
Checks syntax and import validation for all project modules.
Ensures no missing dependencies or syntax errors before launch.
"""

import sys
import os
import importlib

# Add root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

print(f"Project Root: {ROOT_DIR}")
print("=" * 60)
print("  SHIELD-RYZEN V2 INTEGRITY CHECK")
print("=" * 60)

modules_to_check = [
    "shield_types",
    "shield_utils_core",
    "shield_camera",
    "shield_face_pipeline",
    "shield_xception",
    "shield_engine",
    "v3_xdna_engine",
    "shield_hud",
    "shield_hardware_monitor",
    "shield_audio",
    "shield_gradcam",
    "start_shield",
    "export_onnx",
    "quantize_ryzen",
    "validate_system",
    "security.audit_trail",
    "plugins.arcface_reid",
]

failed = []
passed = []

for mod_name in modules_to_check:
    try:
        print(f"Checking {mod_name}...", end=" ")
        importlib.import_module(mod_name)
        print("OK")
        passed.append(mod_name)
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append((mod_name, str(e)))

print("-" * 60)
print(f"Verified: {len(passed)}/{len(modules_to_check)}")

if failed:
    print("\n❌ FOUND ERRORS:")
    for mod, err in failed:
        print(f"  - {mod}: {err}")
    sys.exit(1)
else:
    print("\n✅ ALL MODULES VALID. READY FOR LAUNCH.")
    sys.exit(0)
