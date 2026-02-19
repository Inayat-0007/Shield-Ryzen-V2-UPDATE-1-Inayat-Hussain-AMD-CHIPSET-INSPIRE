"""
Shield-Ryzen V2 — Launcher (TASK 14)
====================================
Main entry point for the Shield-Ryzen Real-Time Deepfake Detection System.
Launches the engine with optimized settings for AMD hardware.
Fullscreen HUD with clean exit.

Usage:
  python start_shield.py --source 0 --cpu
  python start_shield.py --source 1 --cpu    (use second webcam)

Developer: Inayat Hussain | AMD Slingshot 2026
Part 14 of 14 — Final Execution
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

# Add root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from v3_xdna_engine import RyzenXDNAEngine, ShieldEngine, DEFAULT_CONFIG
from shield_hud import ShieldHUD


WINDOW_NAME = "Shield-Ryzen V2 | AMD NPU Secured"


def main():
    parser = argparse.ArgumentParser(description="Shield-Ryzen V2 Launcher")
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0, 1, etc.) or Video File Path")
    parser.add_argument("--model", type=str, default="models/shield_ryzen_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--audit", action="store_true", help="Enable Audit Trail logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (disable NPU Optimization)")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--windowed", action="store_true", help="Run in windowed mode (default is fullscreen)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default 720)")

    args = parser.parse_args()

    # Configure
    config = DEFAULT_CONFIG.copy()

    # Source
    if args.source.isdigit():
        config["camera_id"] = int(args.source)
    else:
        config["camera_id"] = args.source

    # Camera resolution
    config["camera_width"] = args.width
    config["camera_height"] = args.height

    # Model
    config["model_path"] = args.model

    # Audit
    if args.audit:
        config["log_path"] = "logs/shield_audit_session.jsonl"

    # Engine Selection
    if args.cpu:
        print("[SHIELD] Force CPU Mode: Using Standard Engine")
        engine_cls = ShieldEngine
    else:
        print("[SHIELD] NPU Mode: Using RyzenXDNAEngine")
        engine_cls = RyzenXDNAEngine

    # Start
    print("=" * 60)
    print(f"  Shield-Ryzen V2 — Starting...")
    print(f"  Source: {config['camera_id']}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Model:  {config['model_path']}")
    print(f"  Engine: {engine_cls.__name__}")
    print(f"  Mode:   {'Windowed' if args.windowed else 'Fullscreen'}")
    print("=" * 60)

    hud = ShieldHUD()
    engine = None
    exit_requested = False

    try:
        engine = engine_cls(config)

        # ── Setup display window ──
        if not args.headless:
            if args.windowed:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            else:
                # Fullscreen
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Start engine threads
        engine.start()

        print("[SHIELD] System Active. Press 'Q' or 'ESC' to exit.")

        while engine.running and not exit_requested:
            # ── KEY INPUT FIRST (prevents hang) ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\n[SHIELD] Exit key pressed — shutting down...")
                exit_requested = True
                break

            # ── FETCH & RENDER ──
            result = engine.get_latest_result()

            if result and result.frame is not None:
                annotated_frame, _ = hud.render(result.frame, result)

                if not args.headless:
                    cv2.imshow(WINDOW_NAME, annotated_frame)
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[SHIELD] Interrupted by User.")
    except Exception as e:
        import traceback
        err_msg = f"\n[SHIELD] Critical Error: {e}\n{traceback.format_exc()}"
        print(err_msg)
        try:
            with open("crash_log.txt", "w") as f:
                f.write(err_msg)
        except Exception:
            pass
    finally:
        # ── CLEAN EXIT ──
        print("[SHIELD] Cleaning up...")

        # 1. Signal engine to stop
        if engine:
            engine.running = False

        # 2. Destroy windows (must be on main thread)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

        # 3. Join threads + cleanup
        if engine:
            try:
                engine.stop()
            except Exception as e:
                print(f"[SHIELD] Cleanup warning: {e}")

        print("[SHIELD] Shutdown Complete.")


if __name__ == "__main__":
    main()
