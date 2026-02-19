"""
Shield-Ryzen V2 — Ryzen AI NPU Quantizer (TASK 9.3)
===================================================
Produces an optimized INT8 Quantized-DeQuantized (QDQ) ONNX model
targeting AMD Ryzen AI NPUs via Vitis AI Execution Provider.
Uses static quantization with a calibration dataset.

Prerequisites:
  - 'models/shield_ryzen.onnx' (FP32 source)
  - 'data/calibration/' (Calibration images: ~100 real/fake faces)
  - onnxruntime-quantization

Developer: Inayat Hussain | AMD Slingshot 2026
Part 9 of 14 — AMD Native Optimization
"""

import os
import sys
import glob
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shield_face_pipeline import ShieldFacePipeline, _NORM_MEAN, _NORM_STD

class ShieldCalibrationDataReader(CalibrationDataReader):
    """
    Reads calibration images and preprocesses them EXACTLY like the inference pipeline.
    This ensures quantized activations match real deployment distribution.
    """
    def __init__(self, calibration_dir: str, input_name: str = "input_1"):
        self.input_name = input_name
        self.image_paths = glob.glob(os.path.join(calibration_dir, "*.jpg")) + \
                           glob.glob(os.path.join(calibration_dir, "*.png"))
        self.enum_data_dicts = iter([])
        self.pipeline = ShieldFacePipeline(detector_type="mediapipe") # Use MP for consistent crop
        
        print(f"[{len(self.image_paths)}] calibration images found.")
        self.preprocess_all()

    def preprocess_all(self):
        """Pre-load and process all images."""
        data_list = []
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Detect face & align
            # Use pipeline to get crop
            faces = self.pipeline.detect_faces(img)
            if not faces:
                # If image is already a crop? 
                # Try center crop 299x299 if no face
                h, w = img.shape[:2]
                if h >= 299 and w >= 299:
                    cy, cx = h//2, w//2
                    crop = img[cy-149:cy+150, cx-149:cx+150] # 299
                    # Resize to 299 just in case
                    crop_resized = cv2.resize(img, (299, 299)) 
                    # Normalize
                    blob = self._normalize(crop_resized)
                    data_list.append({self.input_name: blob})
                continue
                
            # Take first face
            face = faces[0]
            # face.face_crop_299 is already processed!
            blob = face.face_crop_299
            data_list.append({self.input_name: blob})
            
        self.enum_data_dicts = iter(data_list)
        print(f"[{len(data_list)}] valid calibration samples prepared.")

    def _normalize(self, img_bgr):
        """Duplicate logic from ShieldFacePipeline just in case fallback used."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_float - _NORM_MEAN) / _NORM_STD
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0).astype(np.float32)

    def get_next(self):
        return next(self.enum_data_dicts, None)


def quantize_model(input_model_path, output_model_path, calibration_dir):
    print(f"Quantizing {input_model_path} -> {output_model_path}...")
    
    # 1. Determine Input Name
    model = onnx.load(input_model_path)
    input_name = model.graph.input[0].name
    print(f"Detected Model Input Name: {input_name}")

    # 2. Prepare Calibration Reader
    if not os.path.exists(calibration_dir):
        print(f"Warning: Calibration dir {calibration_dir} missing.")
        os.makedirs(calibration_dir, exist_ok=True)
        # Create dummy image
        dummy = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(calibration_dir, "dummy_calib.jpg"), dummy)
        print("Created dummy calibration image.")

    dr = ShieldCalibrationDataReader(calibration_dir, input_name=input_name)


    if not os.path.exists(calibration_dir) or len(dr.image_paths) == 0:
         print("No calibration images! Quantization requires data.")
         return

    # 2. Quantize
    # QDQ format is preferred for NPU / Vitis AI / TensorRT
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ, # Quantized-DeQuantized format
        per_channel=True,             # Accuracy benefit for Conv weight
        weight_type=QuantType.QInt8,  # Int8 weights
        activation_type=QuantType.QInt8 # Int8 activations
    ) # optimize_model=False to keep nodes standard

    print(f"Quantization Complete. Saved to {output_model_path}")
    print("Verify with Netron or benchmark script.")

if __name__ == "__main__":
    IN_MODEL = "models/shield_ryzen.onnx"
    OUT_MODEL = "models/shield_ryzen_int8.onnx"
    CALIB_DIR = "data/calibration"
    
    if os.path.exists(IN_MODEL):
        quantize_model(IN_MODEL, OUT_MODEL, CALIB_DIR)
    else:
        print(f"Source model {IN_MODEL} not found. Skipping quantization.")
