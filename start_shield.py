"""
Shield-Ryzen V2 — Launcher (TASK 14)
====================================
Main entry point for the Shield-Ryzen Real-Time Deepfake Detection System.
Launches the engine with optimized settings for AMD hardware.
Fullscreen HUD with clean exit.

Usage:
  python start_shield.py --cpu                    (use config.yaml camera_id)
  python start_shield.py --source 1 --cpu         (use plug-in camera)
  python start_shield.py --cpu --windowed         (resizable window)
  python start_shield.py --cpu --size 1280x720    (custom window size)

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

# Load config.yaml for defaults
try:
    import yaml
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _yaml_config = yaml.safe_load(_f)
    _DEFAULT_CAMERA = str(_yaml_config.get('camera_id', 0))
except Exception:
    _yaml_config = {}
    _DEFAULT_CAMERA = "0"

WINDOW_NAME = "Shield-Ryzen V2 | AMD NPU Secured"

# Preset window sizes
WINDOW_PRESETS = {
    "small":  (960, 540),
    "medium": (1280, 720),
    "large":  (1600, 900),
    "hd":     (1920, 1080),
}


def main():
    parser = argparse.ArgumentParser(description="Shield-Ryzen V2 Launcher")
    parser.add_argument("--source", type=str, default=_DEFAULT_CAMERA,
                        help=f"Camera ID (0=built-in, 1=plug-in, etc.) or Video File Path (default: {_DEFAULT_CAMERA} from config.yaml)")
    parser.add_argument("--model", type=str, default="models/shield_ryzen_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--audit", action="store_true", help="Enable Audit Trail logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (disable NPU Optimization)")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--windowed", action="store_true", help="Run in windowed mode (resizable, default is fullscreen)")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height (default 720)")
    parser.add_argument("--size", type=str, default=None,
                        help="Window size: WxH (e.g. 960x540) or preset (small/medium/large/hd)")

    args = parser.parse_args()

    # Configure
    config = DEFAULT_CONFIG.copy()

    # Source — respect config.yaml camera_id as default
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

    # Parse window size
    win_w, win_h = 1280, 720  # Default
    if args.size:
        if args.size.lower() in WINDOW_PRESETS:
            win_w, win_h = WINDOW_PRESETS[args.size.lower()]
        elif 'x' in args.size.lower():
            try:
                parts = args.size.lower().split('x')
                win_w, win_h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                print(f"[SHIELD] Invalid size '{args.size}', using 1280x720")

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
    if args.windowed:
        print(f"  Window: {win_w}x{win_h} (resizable)")
    print("=" * 60)

    hud = ShieldHUD()
    engine = None
    exit_requested = False
    is_fullscreen = not args.windowed

    try:
        engine = engine_cls(config)

        # ── Setup display window ──
        if not args.headless:
            # Always use WINDOW_NORMAL for resizability
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            if args.windowed:
                # Resizable windowed mode
                cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
            else:
                # Fullscreen
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Start engine threads
        engine.start()

        print("[SHIELD] System Active. Press 'Q' or 'ESC' to exit.")
        print("[SHIELD] Keys: F=Toggle Fullscreen | +/-=Resize | 1-4=Preset Sizes")

        while engine.running and not exit_requested:
            # ── KEY INPUT FIRST (prevents hang) ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\n[SHIELD] Exit key pressed — shutting down...")
                exit_requested = True
                break

            # Fullscreen toggle (F key)
            elif key == ord('f') or key == ord('F'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("[SHIELD] Fullscreen ON")
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Windowed ({win_w}x{win_h})")

            # Resize shortcuts
            elif key == ord('+') or key == ord('='):
                win_w = min(win_w + 160, 1920)
                win_h = min(win_h + 90, 1080)
                if not is_fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Resize: {win_w}x{win_h}")
            elif key == ord('-') or key == ord('_'):
                win_w = max(win_w - 160, 640)
                win_h = max(win_h - 90, 360)
                if not is_fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Resize: {win_w}x{win_h}")

            # Preset size shortcuts (1-4)
            elif key == ord('1'):
                win_w, win_h = 960, 540
                if not is_fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Preset: Small ({win_w}x{win_h})")
            elif key == ord('2'):
                win_w, win_h = 1280, 720
                if not is_fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Preset: Medium ({win_w}x{win_h})")
            elif key == ord('3'):
                win_w, win_h = 1600, 900
                if not is_fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Preset: Large ({win_w}x{win_h})")
            elif key == ord('4'):
                win_w, win_h = 1920, 1080
                if not is_fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
                    print(f"[SHIELD] Preset: HD ({win_w}x{win_h})")

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

