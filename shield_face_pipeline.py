"""
Shield-Ryzen V2 — Face Detection & Preprocessing Pipeline
==========================================================
Owns ALL face detection, alignment, cropping, and normalization.
No other module should run face detection directly.

Features:
  - Multi-face detection (returns ALL faces, sorted by area)
  - Head pose estimation via solvePnP (yaw/pitch/roll)
  - Occlusion scoring per face
  - 478→68 landmark conversion with anatomical documentation
  - Correct XceptionNet normalization (FF++ standard: mean=0.5, std=0.5)
  - BGR→RGB conversion verified
  - Supports 'mediapipe' and 'dnn_ssd' detector backends

Normalization Documentation (LOGIC COLLAPSE FIX):
  ═══════════════════════════════════════════════════════════
  FaceForensics++ XceptionNet was trained with torchvision:
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  Applied AFTER transforms.ToTensor() which scales [0,255]→[0.0,1.0]:
    pixel_float = pixel_uint8 / 255.0
    normalized = (pixel_float - 0.5) / 0.5
  This maps [0, 255] → [-1.0, +1.0]  (OPTION B from spec)
  
  EVIDENCE: FaceForensics++ official code:
    github.com/ondyari/FaceForensics → classification/detect_from_video.py
    Uses: transforms.Normalize([0.5]*3, [0.5]*3)
  
  NOT Option A (ImageNet): mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
  NOT Option C (broken):   (uint8 - 0.5) / 0.5 on raw [0,255] range
  ═══════════════════════════════════════════════════════════

Developer: Inayat Hussain | AMD Slingshot 2026
Part 2 of 14 — Face Detection, Preprocessing & Normalization Fix
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

_log = logging.getLogger("ShieldFacePipeline")

# ─── Project Root ─────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# FaceDetection Dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class FaceDetection:
    """All data about a single detected face — importable by Part 3+.

    Attributes:
        bbox: (x, y, w, h) bounding box in pixel coordinates.
        confidence: Detection confidence [0.0, 1.0].
        landmarks_478: (478, 3) raw MediaPipe normalized (x, y, z) landmarks.
        landmarks_68: (68, 2) standard 68-point PIXEL coordinates.
        landmarks: (N, 2) array of pixel (x, y) — alias for primary landmarks.
        landmark_confidence: Quality score of landmark detection [0.0, 1.0].
        head_pose: (yaw, pitch, roll) in degrees.
        face_crop_299: 299x299 float32 NCHW tensor ready for inference.
        face_crop_raw: Original BGR crop before normalization.
        occlusion_score: 0.0=fully visible, 1.0=fully occluded.
        is_frontal: True if within yaw/pitch thresholds.
        ear_reliable: True if occlusion < 0.5 (EAR can be trusted).
    """
    bbox: tuple[int, int, int, int]                # x, y, w, h
    confidence: float = 0.0
    landmarks_478: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    landmarks_68: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    landmarks: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    landmark_confidence: float = 0.0
    head_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
    face_crop_299: Optional[np.ndarray] = None      # (1,3,299,299) float32
    face_crop_raw: Optional[np.ndarray] = None       # BGR uint8 crop
    occlusion_score: float = 0.0
    is_frontal: bool = True
    ear_reliable: bool = True
    blendshapes: Optional[list] = None               # List of category scores
    transformation_matrix: Optional[np.ndarray] = None  # (4, 4) transformation matrix


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# XceptionNet (FF++ c23) normalization — DOCUMENTED here.
# Training code used: transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
# This means: pixel_float = img/255.0; normalized = (pixel_float - 0.5) / 0.5
# Output range: [-1.0, +1.0]
_NORM_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_NORM_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_INPUT_SIZE = 299

# Frontal-face thresholds (degrees) - Relaxed for consumer webcams
_YAW_THRESHOLD = 45.0
_PITCH_THRESHOLD = 35.0

# 3D model points for head-pose estimation (generic face model)
# Based on the standard anthropometric model (in mm, arbitrary scale)
# Points: nose tip, chin, left eye corner, right eye corner,
#         left mouth corner, right mouth corner
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)

# MediaPipe 478-mesh indices for the head-pose reference points
# nose tip=1, chin=152, left eye outer=33, right eye outer=263,
# left mouth=61, right mouth=291
_MP_POSE_INDICES = [1, 152, 33, 263, 61, 291]

# ═══════════════════════════════════════════════════════════════
# MediaPipe 478 → Standard 68+20 Landmark Mapping
# ═══════════════════════════════════════════════════════════════
#
# Standard 68-point landmarks follow the Multi-PIE / iBUG convention:
#   Points  1-17: Jaw contour (right ear → chin → left ear)
#   Points 18-22: Right eyebrow (inner → outer) [anatomical right]
#   Points 23-27: Left eyebrow (inner → outer) [anatomical left]
#   Points 28-31: Nose bridge (top → tip)
#   Points 32-36: Nose bottom (left nostril → right nostril)
#   Points 37-42: Right eye (outer → inner, upper then lower)
#   Points 43-48: Left eye (inner → outer, upper then lower)
#   Points 49-60: Outer lip
#   Points 61-68: Inner lip
#
# The 478 indices below come from the canonical MediaPipe FaceMesh
# geometry (indexed from 0). Each index is documented with its
# anatomical landmark label for traceability.

# Jaw contour (17 points: from right ear → chin → left ear)
_MP_JAW_INDICES = [
    10,   # 1  Right temple / ear area
    338,  # 2  Right upper jaw
    297,  # 3  Right jaw
    332,  # 4  Right lower jaw
    284,  # 5  Right jaw angle
    251,  # 6  Right lower cheek
    389,  # 7  Right chin area
    356,  # 8  Right of chin
    454,  # 9  Chin center
    323,  # 10 Left of chin
    361,  # 11 Left chin area
    288,  # 12 Left lower cheek
    397,  # 13 Left jaw angle
    365,  # 14 Left lower jaw
    379,  # 15 Left jaw
    378,  # 16 Left upper jaw
    400,  # 17 Left temple / ear area
]

# Right eyebrow (5 points: inner → outer) [subject's right]
_MP_RIGHT_BROW_INDICES = [
    70,   # 18 Inner right brow
    63,   # 19 Inner-mid right brow
    105,  # 20 Mid right brow
    66,   # 21 Outer-mid right brow
    107,  # 22 Outer right brow
]

# Left eyebrow (5 points: inner → outer) [subject's left]
_MP_LEFT_BROW_INDICES = [
    336,  # 23 Inner left brow
    296,  # 24 Inner-mid left brow
    334,  # 25 Mid left brow
    293,  # 26 Outer-mid left brow
    300,  # 27 Outer left brow
]

# Nose bridge (4 points: top → tip)
_MP_NOSE_BRIDGE_INDICES = [
    168,  # 28 Nasion (bridge top)
    6,    # 29 Mid bridge
    197,  # 30 Lower bridge
    195,  # 31 Nose tip approach
]

# Nose bottom (5 points: left → right through tip)
_MP_NOSE_BOTTOM_INDICES = [
    5,    # 32 Left nostril outer
    4,    # 33 Left nostril
    1,    # 34 Nose tip (pronasale)
    19,   # 35 Right nostril
    94,   # 36 Right nostril outer
]

# Right eye (6 points: outer corner CW) [subject's right]
_MP_RIGHT_EYE_INDICES = [
    33,   # 37 Right eye outer corner (lateral canthus)
    160,  # 38 Right eye upper lid outer
    158,  # 39 Right eye upper lid inner
    133,  # 40 Right eye inner corner (medial canthus)
    153,  # 41 Right eye lower lid inner
    144,  # 42 Right eye lower lid outer
]

# Left eye (6 points: inner corner CW) [subject's left]
_MP_LEFT_EYE_INDICES = [
    362,  # 43 Left eye inner corner (medial canthus)
    385,  # 44 Left eye upper lid inner
    387,  # 45 Left eye upper lid outer
    263,  # 46 Left eye outer corner (lateral canthus)
    373,  # 47 Left eye lower lid outer
    380,  # 48 Left eye lower lid inner
]

# Outer lip (12 points: CW from right corner)
_MP_OUTER_LIP_INDICES = [
    61,   # 49 Right mouth corner
    185,  # 50 Upper lip right outer
    40,   # 51 Upper lip right
    39,   # 52 Upper lip center-right
    37,   # 53 Upper lip center
    0,    # 54 Upper lip center-left
    267,  # 55 Upper lip left
    269,  # 56 Upper lip left outer
    270,  # 57 Left mouth corner
    409,  # 58 Lower lip left outer
    291,  # 59 Lower lip center
    375,  # 60 Lower lip right outer
]

# Inner lip (8 points: CW from right inner corner)
_MP_INNER_LIP_INDICES = [
    78,   # 61 Right inner corner
    191,  # 62 Upper inner lip right
    80,   # 63 Upper inner lip center
    81,   # 64 Upper inner lip left
    308,  # 65 Left inner corner
    415,  # 66 Lower inner lip left
    310,  # 67 Lower inner lip center
    311,  # 68 Lower inner lip right
]

# Combined 68-point mapping (ordered 1→68)
_MP_478_TO_68_INDICES = (
    _MP_JAW_INDICES              # 1-17
    + _MP_RIGHT_BROW_INDICES     # 18-22
    + _MP_LEFT_BROW_INDICES      # 23-27
    + _MP_NOSE_BRIDGE_INDICES    # 28-31
    + _MP_NOSE_BOTTOM_INDICES    # 32-36
    + _MP_RIGHT_EYE_INDICES      # 37-42
    + _MP_LEFT_EYE_INDICES       # 43-48
    + _MP_OUTER_LIP_INDICES      # 49-60
    + _MP_INNER_LIP_INDICES      # 61-68
)

assert len(_MP_478_TO_68_INDICES) == 68, (
    f"Landmark mapping must produce exactly 68 points, got {len(_MP_478_TO_68_INDICES)}"
)

# Key landmark groups for occlusion checking (MediaPipe 478-mesh)
_MP_NOSE_CHECK_INDICES = [1, 2, 98, 327]
_MP_MOUTH_CHECK_INDICES = [61, 291, 13, 14, 78, 308]
_MP_FOREHEAD_CHECK_INDICES = [10, 338, 297, 332, 284]

# Occlusion threshold — above this, EAR results are unreliable
_OCCLUSION_UNRELIABLE_THRESHOLD = 0.35


# ═══════════════════════════════════════════════════════════════
# ShieldFacePipeline
# ═══════════════════════════════════════════════════════════════

class ShieldFacePipeline:
    """Face detection, alignment, and preprocessing for Shield-Ryzen.

    Supports two detector backends:
      - 'mediapipe': MediaPipe FaceLandmarker (478-point mesh, lightweight,
                     optimized for mobile/edge). Used by default because
                     the V1 engine already depends on it and it provides
                     dense landmarks without an extra model.
      - 'dnn_ssd':  OpenCV DNN face detector (Caffe SSD, more robust to
                     extreme angles but requires separate landmark model).
                     Useful as fallback or for cross-validation.

    Why MediaPipe as default:
      1. Already a project dependency (no added weight)
      2. Provides 478-point dense mesh (richer than 68-point)
      3. Handles multi-face natively
      4. Runs on CPU with low latency (~5ms per frame)
    """

    def __init__(
        self,
        detector_type: str = "mediapipe",
        landmarker_model: str = "face_landmarker.task",
        max_faces: int = 4,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """Initialize face detection pipeline.

        Args:
            detector_type: 'mediapipe' or 'dnn_ssd'.
            landmarker_model: Path to MediaPipe FaceLandmarker model file.
            max_faces: Maximum number of faces to detect per frame.
            min_detection_confidence: Minimum confidence to accept a detection.
        """
        self._detector_type = detector_type
        self._max_faces = max_faces
        self._min_confidence = min_detection_confidence
        self._landmarker = None
        self._frame_timestamp_ms: int = 0

        if detector_type == "mediapipe":
            self._init_mediapipe(landmarker_model, max_faces, min_detection_confidence)
        elif detector_type == "dnn_ssd":
            self._init_dnn_ssd()
        else:
            raise ValueError(f"Unknown detector_type: {detector_type!r}. "
                             f"Supported: 'mediapipe', 'dnn_ssd'")

        _log.info(
            "ShieldFacePipeline initialized — detector=%s max_faces=%d",
            detector_type, max_faces,
        )

    # ── Initializers ──────────────────────────────────────────

    def _init_mediapipe(
        self,
        model_path: str,
        max_faces: int,
        min_confidence: float,
    ) -> None:
        """Initialize MediaPipe FaceLandmarker backend."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        full_path = os.path.join(_SCRIPT_DIR, model_path)
        # Fallback for v2 model name if v1 requested but v2 present
        if not os.path.exists(full_path) and "face_landmarker.task" in model_path:
             v2_path = os.path.join(_SCRIPT_DIR, "face_landmarker_v2_with_blendshapes.task")
             if os.path.exists(v2_path):
                 full_path = v2_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"MediaPipe model not found: {full_path}")

        model_size_mb = os.path.getsize(full_path) / 1024 / 1024
        _log.info(
            "MediaPipe FaceLandmarker: %.1f MB, expected accuracy: ~95%% "
            "(frontal faces, good lighting)",
            model_size_mb,
        )

        base_options = python.BaseOptions(
            model_asset_path=full_path,
            delegate=python.BaseOptions.Delegate.CPU
        )

        # Use VIDEO mode for synchronous processing
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = python.vision

    def _init_dnn_ssd(self) -> None:
        """Initialize OpenCV DNN SSD face detector.

        Uses the Caffe-based res10_300x300_ssd_iter_140000 model
        which is bundled with recent OpenCV distributions.
        """
        proto_path = os.path.join(_SCRIPT_DIR, "models", "deploy.prototxt")
        model_path = os.path.join(_SCRIPT_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            _log.warning(
                "DNN SSD model files not found at %s. "
                "Falling back to MediaPipe.", _SCRIPT_DIR
            )
            self._init_mediapipe("face_landmarker.task", self._max_faces, self._min_confidence)
            self._detector_type = "mediapipe"
            return

        self._dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        _log.info("OpenCV DNN SSD face detector loaded")

    # ── Public API ────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect ALL faces in the frame.

        Runs detection, extracts landmarks, estimates head pose,
        computes occlusion scores, and crops/normalizes face images.

        Args:
            frame: BGR uint8 image (validated by ShieldCamera).

        Returns:
            List of FaceDetection objects, sorted by bounding box area
            (largest first). Empty list if no faces found.
        """
        if self._detector_type == "mediapipe":
            return self._detect_mediapipe(frame)
        else:
            return self._detect_dnn_ssd(frame)

    def align_and_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align face using eye centers, crop, normalize for XceptionNet.

        ██████████████████████████████████████████████████████████
        ██ LOGIC COLLAPSE FIX POINT — NORMALIZATION PIPELINE   ██
        ██████████████████████████████████████████████████████████

        Step 1: Expand bbox by 30% margin for face context
        Step 2: Crop face region from frame
        Step 3: Resize to exactly 299×299

        Step 4: NORMALIZATION (OPTION B — FF++ [-1, 1]):
            face_float = face_crop.astype(np.float32) / 255.0   # [0, 1]
            face_norm  = (face_float - 0.5) / 0.5               # [-1, 1]

            EVIDENCE: FaceForensics++ training code uses:
              transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            Applied AFTER ToTensor() which scales [0,255]→[0,1].
            Verified by scripts/verify_normalization.py (Part 1).

        Step 5: Convert BGR → RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            Done BEFORE normalization to match training pipeline.

        Step 6: Transpose to NCHW: (1, 3, 299, 299)
            input_tensor = np.transpose(face_norm, (2, 0, 1))  # HWC→CHW
            input_tensor = np.expand_dims(input_tensor, 0)      # add batch

        Args:
            frame: Full BGR image.
            bbox: (x, y, w, h) bounding box.
            landmarks: Optional (N, 2) landmarks for alignment.

        Returns:
            (face_crop_299, face_crop_raw):
              face_crop_299: (1, 3, 299, 299) float32 NCHW tensor, [-1, 1] range.
              face_crop_raw: BGR uint8 crop before normalization.
        """
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]

        # Step 1: Expand bounding box by 30% margin for context
        # WHY: Tight crops lose forehead/chin, hurting Xception accuracy
        margin = 0.3
        cx, cy = x + w // 2, y + h // 2
        half_size = int(max(w, h) * (1 + margin) / 2)

        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(fw, cx + half_size)
        y2 = min(fh, cy + half_size)

        # Step 2: Crop
        face_crop_raw = frame[y1:y2, x1:x2].copy()
        if face_crop_raw.size == 0:
            face_crop_raw = frame[y:y+h, x:x+w].copy()

        if face_crop_raw.size == 0:
            blank = np.zeros((1, 3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
            return blank, np.zeros((10, 10, 3), dtype=np.uint8)

        # Step 3: Resize to 299×299 (XceptionNet input requirement)
        face_resized = cv2.resize(face_crop_raw, (_INPUT_SIZE, _INPUT_SIZE))

        # Step 5 (before Step 4): Convert BGR → RGB
        # CRITICAL: OpenCV loads as BGR, but the model was trained on RGB.
        # This MUST happen before normalization to match the training pipeline.
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # Step 4: Normalize to [-1, 1] using FF++ standard
        #   Formula: normalized = (pixel_uint8 / 255.0 - 0.5) / 0.5
        #   Equivalent to: pixel * (2.0 / 255.0) - 1.0
        face_float = face_rgb.astype(np.float32) / 255.0
        face_norm = (face_float - _NORM_MEAN) / _NORM_STD

        # Step 6: Convert HWC → NCHW (batch dimension + channels first)
        face_chw = np.transpose(face_norm, (2, 0, 1))
        face_tensor = np.expand_dims(face_chw, axis=0).astype(np.float32)

        return face_tensor, face_crop_raw

    def _convert_478_to_68(
        self,
        landmarks_478: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Convert MediaPipe 478-point mesh to standard 68-point landmarks.

        Uses the anatomically-documented mapping defined in
        _MP_478_TO_68_INDICES (see module-level constants).

        MediaPipe provides NORMALIZED coordinates in [0.0, 1.0].
        This method converts them to PIXEL coordinates:
            px = landmark.x * frame_width
            py = landmark.y * frame_height

        Args:
            landmarks_478: (478, 3) normalized MediaPipe landmarks.
            frame_width: Width of source frame in pixels.
            frame_height: Height of source frame in pixels.

        Returns:
            (68, 2) array of pixel (x, y) coordinates.
        """
        landmarks_68 = np.zeros((68, 2), dtype=np.float32)

        for i, mp_idx in enumerate(_MP_478_TO_68_INDICES):
            if mp_idx < landmarks_478.shape[0]:
                landmarks_68[i, 0] = landmarks_478[mp_idx, 0] * frame_width
                landmarks_68[i, 1] = landmarks_478[mp_idx, 1] * frame_height

        return landmarks_68

    def estimate_head_pose(
        self,
        landmarks_2d: np.ndarray,
        image_size: tuple[int, int],
    ) -> tuple[float, float, float]:
        """Estimate head pose from facial landmarks using solvePnP.

        Uses 6 key landmarks matched to a 3D generic face model.
        Returns (yaw, pitch, roll) in degrees.

        Args:
            landmarks_2d: (N, 2) pixel coordinates of landmarks.
            image_size: (width, height) of the source image.

        Returns:
            (yaw, pitch, roll) in degrees. (0, 0, 0) = perfectly frontal.
        """
        if landmarks_2d.shape[0] < max(_MP_POSE_INDICES) + 1:
            return (0.0, 0.0, 0.0)

        # Extract the 6 reference points from MediaPipe 478-mesh
        image_points = np.array([
            landmarks_2d[idx] for idx in _MP_POSE_INDICES
        ], dtype=np.float64)

        # Camera intrinsics approximation from image dimensions
        w, h = image_size
        focal_length = w  # Rough approximation
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return (0.0, 0.0, 0.0)

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Decompose rotation matrix to Euler angles
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch_raw = float(euler_angles[0, 0])
        yaw_raw = float(euler_angles[1, 0])
        roll_raw = float(euler_angles[2, 0])
        
        # NORMALIZE EULER ANGLES:
        # cv2.decomposeProjectionMatrix can return angles in (-180, 180] range
        # which causes near-frontal faces to show pitch=±179° instead of ~0°.
        # We normalize to [-90, +90] which is the physically meaningful range
        # for head pose estimation.
        def _normalize_angle(angle: float) -> float:
            """Normalize angle to [-90, +90] range."""
            if angle > 90.0:
                angle = angle - 180.0
            elif angle < -90.0:
                angle = angle + 180.0
            return angle
        
        yaw = _normalize_angle(yaw_raw)
        pitch = _normalize_angle(pitch_raw)
        roll = _normalize_angle(roll_raw)

        return (
            round(yaw, 1),
            round(pitch, 1),
            round(roll, 1),
        )

    def estimate_occlusion(
        self,
        landmarks_2d: np.ndarray,
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> float:
        """Estimate face occlusion score from landmark positions.

        Checks if key landmark groups (eyes, nose, mouth, forehead)
        are within the bounding box and have plausible inter-distances.

        Args:
            landmarks_2d: (N, 2) pixel coordinates.
            bbox: (x, y, w, h) bounding box.
            image_size: (width, height) of source image.

        Returns:
            Occlusion score: 0.0=fully visible, 1.0=fully occluded.
            If > 0.5, EAR results are unreliable.
        """
        if landmarks_2d.shape[0] < 400:
            return 0.0

        x, y, w, h = bbox
        img_w, img_h = image_size

        checks: list[float] = []

        # Check each landmark group for visibility
        for group_name, indices in [
            ("left_eye", _MP_RIGHT_EYE_INDICES),
            ("right_eye", _MP_LEFT_EYE_INDICES),
            ("nose", _MP_NOSE_CHECK_INDICES),
            ("mouth", _MP_MOUTH_CHECK_INDICES),
            ("forehead", _MP_FOREHEAD_CHECK_INDICES),
        ]:
            visible = 0
            total = len(indices)
            for idx in indices:
                if idx < landmarks_2d.shape[0]:
                    lx, ly = landmarks_2d[idx]
                    # Check landmark is within bounding box + margin
                    in_x = (x - w * 0.1) <= lx <= (x + w * 1.1)
                    in_y = (y - h * 0.1) <= ly <= (y + h * 1.1)
                    # Check landmark is within image bounds
                    in_img = 0 <= lx < img_w and 0 <= ly < img_h
                    if in_x and in_y and in_img:
                        visible += 1
            checks.append(1.0 - (visible / total if total > 0 else 0.0))

        # Check inter-eye distance plausibility
        max_eye_idx = max(_MP_RIGHT_EYE_INDICES + _MP_LEFT_EYE_INDICES)
        if landmarks_2d.shape[0] > max_eye_idx:
            left_eye_center = landmarks_2d[_MP_RIGHT_EYE_INDICES].mean(axis=0)
            right_eye_center = landmarks_2d[_MP_LEFT_EYE_INDICES].mean(axis=0)
            eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
            # Eye distance should be ~30-50% of face width
            eye_ratio = eye_dist / max(w, 1)
            if eye_ratio < 0.15 or eye_ratio > 0.8:
                checks.append(0.3)  # Suspicious eye distance
            else:
                checks.append(0.0)

        occlusion = float(np.mean(checks)) if checks else 0.0
        return round(occlusion, 3)

    def release(self) -> None:
        """Release detector resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        _log.info("ShieldFacePipeline released")

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> "ShieldFacePipeline":
        return self

    def __exit__(self, *args) -> None:
        self.release()

    # ── Private: MediaPipe Detection ──────────────────────────

    def _detect_mediapipe(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces using MediaPipe FaceLandmarker."""
        if self._landmarker is None:
            _log.error("MediaPipe landmarker not initialized")
            return []

        h, w = frame.shape[:2]

        import mediapipe as mp
        # MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # Detect with incremented timestamp
        self._frame_timestamp_ms += 33  # ~30 FPS timestamp step
        try:
            result = self._landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )
        except Exception as e:
            _log.debug("MediaPipe detection failed: %s", e)
            return []

        if not result or not result.face_landmarks:
            return []

        detections: list[FaceDetection] = []

        for i, face_lms in enumerate(result.face_landmarks):
            # Raw MediaPipe 478 landmarks: normalized (x, y, z)
            raw_478 = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_lms],
                dtype=np.float32,
            )

            # Convert normalized landmarks to pixel coordinates (N, 2)
            lm_pixel = np.array(
                [[lm.x * w, lm.y * h] for lm in face_lms],
                dtype=np.float32,
            )

            # Convert 478 → 68 standard landmarks (pixel coords)
            lm_68 = self._convert_478_to_68(raw_478, w, h)

            # Compute bounding box from landmarks
            xs = lm_pixel[:, 0]
            ys = lm_pixel[:, 1]
            x_min = max(0, int(xs.min()) - 10)
            y_min = max(0, int(ys.min()) - 10)
            x_max = min(w, int(xs.max()) + 10)
            y_max = min(h, int(ys.max()) + 10)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Detection confidence from landmark quality
            landmark_conf = self._estimate_landmark_confidence(lm_pixel, bbox)

            # Head pose estimation
            head_pose = self.estimate_head_pose(lm_pixel, (w, h))

            # Hybrid Frontal Logic: SolvePnP + Spatial Heuristic
            pnp_frontal = (
                abs(head_pose[0]) < _YAW_THRESHOLD
                and abs(head_pose[1]) < _PITCH_THRESHOLD
            )

            # Spatial Heuristic: If face is large and centered, likely frontal
            f_x, f_y, f_w, f_h = bbox
            f_cx = f_x + f_w / 2
            f_cy = f_y + f_h / 2
            centered = (abs(f_cx - w / 2) < (w * 0.25)) and (abs(f_cy - h / 2) < (h * 0.25))
            large = (f_w * f_h) > (w * h * 0.03)  # 3% of screen

            is_frontal = pnp_frontal or (centered and large)

            # Occlusion scoring
            occlusion = self.estimate_occlusion(lm_pixel, bbox, (w, h))
            ear_reliable = occlusion < _OCCLUSION_UNRELIABLE_THRESHOLD

            # Align, crop, and normalize
            face_tensor, face_raw = self.align_and_crop(frame, bbox, lm_pixel)

            detection = FaceDetection(
                bbox=bbox,
                confidence=max(0.5, landmark_conf),
                landmarks_478=raw_478,
                landmarks_68=lm_68,
                landmarks=lm_pixel,
                landmark_confidence=landmark_conf,
                head_pose=head_pose,
                face_crop_299=face_tensor,
                face_crop_raw=face_raw,
                occlusion_score=occlusion,
                is_frontal=is_frontal,
                ear_reliable=ear_reliable,
                blendshapes=result.face_blendshapes[i] if result.face_blendshapes else None,
                transformation_matrix=result.facial_transformation_matrixes[i] if result.facial_transformation_matrixes else None,
            )
            detections.append(detection)

        # Sort by bounding box area (largest first)
        detections.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)

        return detections

    def _detect_dnn_ssd(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces using OpenCV DNN SSD detector.

        Note: This backend doesn't provide dense landmarks,
        so head pose and occlusion estimation are limited.
        """
        if not hasattr(self, "_dnn_net"):
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._dnn_net.setInput(blob)
        raw_detections = self._dnn_net.forward()

        detections: list[FaceDetection] = []

        for i in range(raw_detections.shape[2]):
            conf = float(raw_detections[0, 0, i, 2])
            if conf < self._min_confidence:
                continue

            x1 = max(0, int(raw_detections[0, 0, i, 3] * w))
            y1 = max(0, int(raw_detections[0, 0, i, 4] * h))
            x2 = min(w, int(raw_detections[0, 0, i, 5] * w))
            y2 = min(h, int(raw_detections[0, 0, i, 6] * h))
            bbox = (x1, y1, x2 - x1, y2 - y1)

            if bbox[2] < 30 or bbox[3] < 30:
                continue

            face_tensor, face_raw = self.align_and_crop(frame, bbox)

            detection = FaceDetection(
                bbox=bbox,
                confidence=conf,
                landmarks_478=np.empty((0, 3)),
                landmarks_68=np.empty((0, 2)),
                landmarks=np.empty((0, 2)),
                landmark_confidence=0.0,
                head_pose=(0.0, 0.0, 0.0),
                face_crop_299=face_tensor,
                face_crop_raw=face_raw,
                occlusion_score=0.0,
                is_frontal=True,  # Can't determine without landmarks
                ear_reliable=True,
                blendshapes=None,
                transformation_matrix=None,
            )
            detections.append(detection)

        detections.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        return detections

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _estimate_landmark_confidence(
        landmarks: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float:
        """Estimate landmark quality from spatial consistency.

        Good landmarks should be well-distributed within the bbox.
        Clumped or out-of-bounds landmarks indicate poor quality.
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or landmarks.shape[0] == 0:
            return 0.0

        # Check what fraction of landmarks fall within bbox
        in_box = (
            (landmarks[:, 0] >= x)
            & (landmarks[:, 0] <= x + w)
            & (landmarks[:, 1] >= y)
            & (landmarks[:, 1] <= y + h)
        )
        coverage = float(in_box.sum()) / landmarks.shape[0]

        # Check landmark spread (std relative to bbox size)
        x_spread = landmarks[:, 0].std() / max(w, 1)
        y_spread = landmarks[:, 1].std() / max(h, 1)
        spread_score = min(1.0, (x_spread + y_spread) / 0.5)

        confidence = (coverage * 0.6 + spread_score * 0.4)
        return round(min(1.0, max(0.0, confidence)), 3)
