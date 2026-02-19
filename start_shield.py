"""
Shield-Ryzen V2 — Launcher (TASK 14)
====================================
Main entry point for the Shield-Ryzen Real-Time Deepfake Detection System.
Launches the engine with optimized settings for AMD hardware and displays the HUD.

Usage:
  python start_shield.py --source 0 --model models/shield_ryzen_int8.onnx [--audit]

Developer: Inayat Hussain | AMD Slingshot 2026
Part 14 of 14 — Final Execution
"""

import argparse
import sys
import os
import signal
import time
import cv2
import numpy as np

# Add root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from v3_xdna_engine import RyzenXDNAEngine, ShieldEngine, DEFAULT_CONFIG
from shield_hud import ShieldHUD

def signal_handler(sig, frame):
    print("\n[SHIELD] Shutdown signal received based on user interrupt.")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Shield-Ryzen V2 Launcher")
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0) or Video File Path")
    parser.add_argument("--model", type=str, default="models/shield_ryzen_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--audit", action="store_true", help="Enable Audit Trail logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (disable NPU Optimization)")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    
    args = parser.parse_args()
    
    # Configure
    config = DEFAULT_CONFIG.copy()
    
    # Source
    if args.source.isdigit():
        config["camera_id"] = int(args.source)
    else:
        config["camera_id"] = args.source
        
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
    print(f"  Model:  {config['model_path']}")
    print(f"  Engine: {engine_cls.__name__}")
    print("=" * 60)
    
    hud = ShieldHUD()
    engine = None
    
    try:
        engine = engine_cls(config)
        
        # Register signal
        signal.signal(signal.SIGINT, signal_handler)
        
        engine.start()
        
        # Main Loop
        print("[SHIELD] System Active. Press 'Q' or 'ESC' to exit.")
        while engine.running:
            # Fetch latest result
            result = engine.get_latest_result()
            
            if result and result.frame is not None:
                # Render HUD
                annotated_frame, _ = hud.render(result.frame, result)
                
                if not args.headless:
                    cv2.imshow("Shield-Ryzen V2 | AMD NPU Secured", annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27: # Q or ESC
                        break
            else:
                # Idle wait if no frame ready
                time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n[SHIELD] Interrupted by User.")
    except Exception as e:
        import traceback
        err_msg = f"\n[SHIELD] Critical Error: {e}\n{traceback.format_exc()}"
        print(err_msg)
        with open("crash_log.txt", "w") as f:
            f.write(err_msg)
    finally:
        if engine:
            engine.stop()
        cv2.destroyAllWindows()
        print("[SHIELD] Shutdown Complete.")

if __name__ == "__main__":
    main()
