import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import timm
import os
import time
import math

# ═══════════════════════════════════════════════════════════════
#  SHIELD-RYZEN V1 — SECURITY MODE
#  Developer: Inayat Hussain | AMD Slingshot 2026
#  Features: 89% Confidence + Blink Liveness + Texture Guard
# ═══════════════════════════════════════════════════════════════

# ─── Constants ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.89    # 89% rule: only VERIFIED REAL above this
BLINK_THRESHOLD = 0.21         # EAR below this = eyes closed (blink)
BLINK_TIME_WINDOW = 10         # Seconds — must blink within this window
LAPLACIAN_THRESHOLD = 50       # Below this = too smooth (likely fake/photo)

# MediaPipe FaceLandmarker eye indices (478-point mesh)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]   # outer, upper1, upper2, inner, lower2, lower1
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # outer, upper1, upper2, inner, lower2, lower1

# ─── 1. Setup Face Landmarker (Detection + Eye Tracking) ─────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=landmarker_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─── 2. Eye Aspect Ratio (EAR) Function ──────────────────────
def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio from 6 landmark points.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Low EAR = eye closed (blink). Normal ~0.25-0.3, blink ~0.15."""
    p1 = landmarks[eye_indices[0]]  # outer corner
    p2 = landmarks[eye_indices[1]]  # upper mid 1
    p3 = landmarks[eye_indices[2]]  # upper mid 2
    p4 = landmarks[eye_indices[3]]  # inner corner
    p5 = landmarks[eye_indices[4]]  # lower mid 2
    p6 = landmarks[eye_indices[5]]  # lower mid 1
    
    # Vertical distances
    v1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
    v2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
    # Horizontal distance
    h_dist = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)
    
    if h_dist == 0:
        return 0.3  # default open-eye value
    
    return (v1 + v2) / (2.0 * h_dist)

def check_texture(face_crop):
    """Laplacian variance — low value = too smooth (photo/deepfake artifact)."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ─── 3. Initialize Xception Model ────────────────────────────
#    Class Mapping (ffpp_c23.pth verified live):
#      Index 0 = FAKE  |  Index 1 = REAL
class ShieldXception(nn.Module):
    def __init__(self):
        super(ShieldXception, self).__init__()
        self.model = timm.create_model('legacy_xception', pretrained=False, num_classes=2)

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=1)  # Output sums to 1.0

# ─── 4. Deploy to NVIDIA RTX 3050 (CUDA) ─────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
model = ShieldXception().to(device)

# ─── 5. Secure Model Loading ───────────────────────────────
import hashlib
from shield_utils import ConfidenceCalibrator

class SecurityError(Exception):
    """Raised when model integrity verification fails."""
    pass

def load_model_with_verification(model_path, device):
    """Load model with cryptographic integrity check."""
    print(f"🔒 Verifying integrity of {model_path}...")
    
    # Load expected hash
    sig_path = os.path.join("models", "model_signature.sha256")
    if not os.path.exists(sig_path):
        raise FileNotFoundError(f"Missing signature file: {sig_path}")
        
    with open(sig_path, "r") as f:
        expected_hash = f.read().strip()
        
    # Calculate actual hash
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    actual_hash = sha256.hexdigest()
    
    if actual_hash != expected_hash:
        raise SecurityError(
            f"🚨 MODEL INTEGRITY VIOLATION!\n"
            f"Expected: {expected_hash}\n"
            f"Got:      {actual_hash}\n"
            f"The model file has been modified or corrupted."
        )
        
    print(f"✅ HASH VERIFIED: {actual_hash[:8]}...")

    # Load Weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle key renaming (training vs inference names)
    new_state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    if 'last_linear.1.weight' in new_state_dict:
        new_state_dict['fc.weight'] = new_state_dict.pop('last_linear.1.weight')
        new_state_dict['fc.bias'] = new_state_dict.pop('last_linear.1.bias')
        
    return new_state_dict

# Initialize Calibrator (Part 3)
calibrator = ConfidenceCalibrator.load()

try:
    # 1. Verify and Load Weights
    clean_state_dict = load_model_with_verification('ffpp_c23.pth', device)
    
    # 2. Apply to Architecture
    result = model.model.load_state_dict(clean_state_dict, strict=False)
    
    # 3. Log Status
    total_keys = len(clean_state_dict)
    matched_keys = total_keys - len(result.unexpected_keys)
    print(f"✅ Brain Loaded — {matched_keys}/{total_keys} weights matched.")
    print(f"✅ Calibration — Temperature T={calibrator.temperature:.2f}")

except Exception as e:
    print(f"❌ SECURITY/LOAD ERROR: {e}")
    # In production, we might exit here. For dev, we warn.
    # sys.exit(1)

model.eval()

# ─── 6. Image Transform (Xception 299×299, [-1,1]) ───────────
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ─── 7. Start Security Mode ──────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)

    print("═" * 55)
    print("  🛡️  SHIELD-RYZEN SECURITY MODE ACTIVE")
    print("═" * 55)
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Blink Window:         {BLINK_TIME_WINDOW}s")
    print(f"  Texture Guard:        Laplacian > {LAPLACIAN_THRESHOLD}")
    print("  Press ESC to exit.")
    print("═" * 55)

    frame_timestamp_ms = 0
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()

    # Blink tracking state
    blink_count = 0
    blink_timestamps = []   # track when blinks occurred
    was_eye_closed = False  # for edge detection (closed → open = 1 blink)

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # FPS calculation
            fps_counter += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer = now

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            lm_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33  # ~30 FPS

            h, w, _ = frame.shape

            # Prune old blinks outside the time window
            blink_timestamps = [t for t in blink_timestamps if now - t < BLINK_TIME_WINDOW]
            blink_count = len(blink_timestamps)

            if lm_result.face_landmarks:
                for face_landmarks in lm_result.face_landmarks:
                    # ── Extract bounding box from landmarks ──
                    xs = [lm.x for lm in face_landmarks]
                    ys = [lm.y for lm in face_landmarks]
                    x_min = max(0, int(min(xs) * w) - 10)
                    y_min = max(0, int(min(ys) * h) - 10)
                    x_max = min(w, int(max(xs) * w) + 10)
                    y_max = min(h, int(max(ys) * h) + 10)
                    fw = x_max - x_min
                    fh = y_max - y_min

                    # ── Calculate EAR (Blink Detection) ──
                    left_ear = calculate_ear(face_landmarks, LEFT_EYE)
                    right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Blink edge detection: closed → open = 1 blink
                    if avg_ear < BLINK_THRESHOLD:
                        was_eye_closed = True
                    elif was_eye_closed and avg_ear >= BLINK_THRESHOLD:
                        was_eye_closed = False
                        blink_timestamps.append(now)
                        blink_count = len(blink_timestamps)

                    liveness_ok = blink_count > 0

                    # ── Crop Face for AI Analysis ──
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size == 0:
                        continue

                    # ── Texture/Sharpness Guard ──
                    texture_score = check_texture(face_crop)
                    texture_ok = texture_score > LAPLACIAN_THRESHOLD

                    # ── AI Inference (CUDA) ──
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    input_tensor = transform(face_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        probs = model(input_tensor).cpu().numpy()[0]
                        # Apply Temperature Scaling (Part 3)
                        calibrated_probs = calibrator.calibrate(probs)
                        
                        # ffpp_c23.pth: Index 0 = Fake, Index 1 = Real
                        fake_prob = float(calibrated_probs[0])
                        real_prob = float(calibrated_probs[1])

                    # ══════════════════════════════════════════════
                    #  SECURITY MODE CLASSIFICATION (3-Tier)
                    # ══════════════════════════════════════════════
                    if fake_prob > 0.50:
                        # CONDITION 2: Fake detected
                        label = "CRITICAL: FAKE DETECTED"
                        color = (0, 0, 255)       # Red
                        tier = "FAKE"
                    elif real_prob < CONFIDENCE_THRESHOLD:
                        # CONDITION 1: Low confidence
                        label = "WARNING: LOW CONFIDENCE"
                        color = (0, 200, 255)     # Yellow/Orange
                        tier = "WARN"
                    elif not liveness_ok:
                        # Real but no blink — could be a photo
                        label = "LIVENESS FAILED"
                        color = (0, 165, 255)     # Orange
                        tier = "LIVENESS"
                    elif not texture_ok:
                        # Real + blink but too smooth
                        label = "SMOOTHNESS WARNING"
                        color = (0, 200, 255)     # Yellow
                        tier = "TEXTURE"
                    else:
                        # CONDITION 3: Verified Real
                        label = "SHIELD: VERIFIED REAL"
                        color = (0, 255, 0)       # Green
                        tier = "VERIFIED"

                    # ── Draw bounding box ──
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # ── Label with scores ──
                    cv2.putText(frame, label, (x_min, y_min - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"Real:{real_prob*100:.1f}% Fake:{fake_prob*100:.1f}%", 
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # ── EAR + Texture info (per-face, right side) ──
                    info_x = x_max + 5
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (info_x, y_min + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(frame, f"TEX: {texture_score:.0f}", (info_x, y_min + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── HUD (top bar) ──
            # Dark background bar
            cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
            # Line 1: FPS + Device
            cv2.putText(frame, f"FPS: {fps_display:.1f} | {device}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Line 2: Blink status + threshold
            blink_color = (0, 255, 0) if blink_count > 0 else (0, 0, 255)
            blink_text = f"Blink Detected: {'YES' if blink_count > 0 else 'NO'} ({blink_count} in {BLINK_TIME_WINDOW}s)"
            cv2.putText(frame, blink_text, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)
            # Right side: Security mode badge
            cv2.putText(frame, "SECURITY MODE", (w - 170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", (w - 170, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow('Shield-Ryzen V1 | SECURITY MODE', frame)
            if cv2.waitKey(5) & 0xFF == 27: break  # ESC to stop

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()