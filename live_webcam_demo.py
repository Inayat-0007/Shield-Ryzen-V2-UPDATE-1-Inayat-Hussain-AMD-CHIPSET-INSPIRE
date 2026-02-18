
import cv2
import yaml
import os
import sys
import time
from v3_int8_engine import ShieldEngine

def run_live():
    # Load config
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    # Ensure plug-in webcam is used (Default 0 for safety)
    config["camera_id"] = 0
    # Use the optimized INT8 model
    config["model_path"] = "shield_ryzen_int8.onnx"
    
    try:
        print("[SHIELD] Initializing Shield-Ryzen V2 Engine...")
        engine = ShieldEngine(config)
        print("[OK] Engine Ready. Opening Webcam Window...")
        print(">> Press 'q' to quit.")
        
        # Open window
        cv2.namedWindow("Shield-Ryzen V2 Live", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while True:
            result = engine.process_frame()
            
            if result.frame is not None:
                # Show frame with HUD
                cv2.imshow("Shield-Ryzen V2 Live", result.frame)
            
            frame_count += 1
            # Console Log (Throttled or removed for performance)
            if result.face_results:
                f = result.face_results[0]
                print(f"[{frame_count:04d}] STATE: {f.state:<10} | NEURAL: {f.neural_confidence:.2f} | FPS: {result.fps:.1f} | EAR: {f.ear_value:.2f} | TEX: {f.texture_score:.2f} | OCC: {f.occlusion_score:.2f}")
            elif frame_count % 30 == 0:
                print(f"[{frame_count:04d}] NO FACE DETECTED (FPS: {result.fps:.1f})")
            
            # Check for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"[ERROR] Error during live run: {e}")
    finally:
        if 'engine' in locals():
            engine.release()
        cv2.destroyAllWindows()
        print("[EXIT] Live Demo Closed.")

if __name__ == "__main__":
    run_live()
