# ============================================================
# SMART LITTERING DETECTION SYSTEM
# ============================================================
# 
# MODELS:
# 1. Person: models/person_detect.pt (person detection)
# 2. Garbage: models/garbage_detect.pt (8 waste classes - OUR TRAINED MODEL)
# 3. Hand: models/hand_landmarker.task (MediaPipe hand landmarks)
# 4. Dustbin: models/dustbin_detect.pt (dustbin detection - OUR TRAINED MODEL)
#
# KEY RULES:
# - Detect garbage near person using OUR custom garbage_detect.pt
# - Stable detection (no flickering - state machine with hysteresis)
# - Grace time: 10 seconds to dispose properly
# - Dustbin bonus: +5 seconds if near dustbin
# - Shows waste CATEGORY + dustbin COLOR for each item
# - Evidence capture on confirmed littering
#
# ============================================================

import cv2
import time
import os
from ultralytics import YOLO

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Import modules (relative to detection/ package)
from detection.dustbin_detection import detect_dustbin, reset_tracker as reset_dustbin
from detection.garbage_detection import get_garbage_detections, reset_tracker as reset_garbage, enable_hand_detection
from detection.hand_detection import detect_hands, reset_tracker as reset_hand, is_model_available as hand_model_available
from detection.logic.littering_decision import LitteringTracker
from detection.utils.evidence_manager import EvidenceManager
from backend.utils import normalize_class_name

# ============================================================
# CONFIGURATION
# ============================================================

# Timing
FPS = 30
GRACE_TIME = 10  # Seconds to dispose garbage
DUSTBIN_BONUS = 5  # Extra time if near dustbin

# Confidence
GARBAGE_CONF = 0.25
PERSON_CONF = 0.40

# Person must be visible for 1 second before detection starts
PERSON_VERIFY_FRAMES = 30

# Evidence
PRE_BUFFER = 5  # Seconds before event
POST_RECORD = 5  # Seconds after event
COOLDOWN = 30  # Seconds between captures

# Waste category -> dustbin color mapping (BGR for OpenCV)
CATEGORY_BGR = {
    "RECYCLABLE": (230, 150, 0),   # Blue
    "ORGANIC":    (0, 180, 0),     # Green
    "HAZARDOUS":  (0, 0, 230),     # Red
    "GENERAL":    (140, 140, 140), # Grey
}
DUSTBIN_NAMES = {
    "RECYCLABLE": "Blue Bin",
    "ORGANIC":    "Green Bin",
    "HAZARDOUS":  "Red Bin",
    "GENERAL":    "Grey Bin",
}

# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 60)
print("🎥 SMART LITTERING DETECTION SYSTEM")
print("=" * 60)
print("Loading models...")

# Person model
person_model_path = os.path.join(PROJECT_ROOT, "models", "person_detect.pt")
person_model = YOLO(person_model_path)
print("✅ Person model: YOLOv8n")

# Pre-load detection models at startup
from detection.dustbin_detection import _load_model as load_dustbin_model
from detection.garbage_detection import _load_model as load_garbage_model
from detection.hand_detection import _load_model as load_hand_model
load_dustbin_model()
load_garbage_model()
load_hand_model()

# Check if hand detection is available and enable it
if hand_model_available():
    enable_hand_detection(True)
    print("✅ Hand detection model loaded - Using hands for precise tracking")
else:
    print("⚠️ Hand detection model not found - Using person body detection")

# Littering logic
littering_tracker = LitteringTracker(
    grace_time=GRACE_TIME,
    near_dustbin_bonus=DUSTBIN_BONUS
)

# Evidence manager
evidence_folder = os.path.join(PROJECT_ROOT, "evidence")
evidence_manager = EvidenceManager(
    base_folder=evidence_folder,
    fps=FPS,
    pre_buffer_seconds=PRE_BUFFER,
    post_record_seconds=POST_RECORD,
    cooldown_seconds=COOLDOWN,
    min_garbage_confidence=GARBAGE_CONF,
    min_person_confidence=PERSON_CONF
)

# State
person_frames = 0
person_verified = False
person_missing_frames = 0

print("=" * 60)
print(f"📊 Garbage confidence: {GARBAGE_CONF}")
print(f"📊 Person confidence: {PERSON_CONF}")
print(f"👤 Person verify: {PERSON_VERIFY_FRAMES} frames ({PERSON_VERIFY_FRAMES/FPS:.1f}s)")
print(f"⏱️ Grace time: {GRACE_TIME}s (+{DUSTBIN_BONUS}s near dustbin)")
print(f"📹 Evidence: {PRE_BUFFER}s before + {POST_RECORD}s after")
print("=" * 60)
print("Controls: 'q' = quit, 'r' = reset")
print("=" * 60 + "\n")

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot open camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
start_time = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    display_frame = frame.copy()
    
    # Buffer for evidence
    evidence_manager.add_frame(frame.copy())
    
    # --------------------------------------------------
    # 1. PERSON DETECTION
    # --------------------------------------------------
    person_results = person_model(frame, verbose=False, conf=PERSON_CONF)
    
    person_boxes = []
    person_detections = []
    
    for result in person_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = person_model.names[cls_id]
            
            if cls_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                person_boxes.append((x1, y1, x2, y2))
                person_detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': conf
                })
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Person {conf:.0%}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # --------------------------------------------------
    # 2. PERSON VERIFICATION
    # --------------------------------------------------
    if person_boxes:
        person_frames += 1
        person_missing_frames = 0
        if person_frames >= PERSON_VERIFY_FRAMES:
            if not person_verified:
                print("✅ Person verified")
            person_verified = True
    else:
        person_missing_frames += 1
        
        if person_missing_frames > 10:
            if person_verified:
                print("👤 Person left frame")
                littering_tracker.reset()
                reset_garbage()
            person_frames = 0
            person_verified = False
    
    # --------------------------------------------------
    # 3. DUSTBIN DETECTION
    # --------------------------------------------------
    display_frame, dustbin_boxes = detect_dustbin(display_frame, confidence_threshold=0.65)
    
    # --------------------------------------------------
    # 3.5 HAND DETECTION
    # --------------------------------------------------
    hand_boxes = []
    if person_verified and hand_model_available():
        hand_boxes = detect_hands(frame, confidence_threshold=0.50, person_boxes=person_boxes)
    
    # --------------------------------------------------
    # 4. GARBAGE DETECTION (only if person verified)
    # --------------------------------------------------
    garbage_boxes = []
    garbage_detections = []
    garbage_in_hands = []
    
    if person_verified:
        garbage_detections = get_garbage_detections(
            frame,
            confidence_threshold=GARBAGE_CONF,
            person_boxes=person_boxes,
            hand_boxes=hand_boxes
        )
        
        for det in garbage_detections:
            x1, y1, x2, y2 = det['box']
            cls_name = det['class_name']
            conf = det['confidence']
            in_hands = det.get('in_hands', False)
            
            garbage_boxes.append((x1, y1, x2, y2))
            garbage_in_hands.append(in_hands)
            
            # Map to waste category and dustbin color
            category = normalize_class_name(cls_name)
            bin_name = DUSTBIN_NAMES.get(category, "Grey Bin")
            cat_color = CATEGORY_BGR.get(category, (140, 140, 140))
            
            if in_hands:
                color = (0, 255, 0)
                label = f"{cls_name} (holding)"
            else:
                color = (0, 0, 255)
                label = f"{cls_name} (dropped!)"
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show waste category + dustbin color below the box
            bin_label = f"{category} -> {bin_name}"
            cv2.putText(display_frame, bin_label, (x1, y2 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cat_color, 2)
    
    # --------------------------------------------------
    # 5. LITTERING LOGIC
    # --------------------------------------------------
    littering_confirmed = False
    status_text = ""
    
    if not person_verified:
        if person_frames > 0:
            pct = int((person_frames / PERSON_VERIFY_FRAMES) * 100)
            status_text = f"Verifying person... {pct}%"
            color = (0, 255, 255)
        else:
            status_text = "Waiting for person..."
            color = (128, 128, 128)
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    elif garbage_boxes and not evidence_manager.is_recording:
        results, triggered, status_text = littering_tracker.check_littering(
            garbage_boxes, dustbin_boxes, person_boxes, garbage_in_hands
        )
        
        for gbox, status in results:
            gx1, gy1, gx2, gy2 = gbox
            
            if status == "HOLDING":
                cv2.putText(display_frame, "OK", (gx1, gy2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif status == "SAFE":
                cv2.putText(display_frame, "SAFE", (gx1, gy2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif status == "WAIT":
                cv2.putText(display_frame, "WAITING...", (gx1, gy2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif status == "WARNING":
                cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), (0, 165, 255), 3)
                cv2.putText(display_frame, "WARNING!", (gx1, gy2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif status == "FINAL_WARNING":
                cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 4)
                cv2.putText(display_frame, "PICK UP NOW!", (gx1, gy2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif status == "LITTERING":
                cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 5)
                cv2.putText(display_frame, "LITTERING!", (gx1, gy2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                littering_confirmed = True
        
        if status_text:
            if "OK" in status_text or "SAFE" in status_text or "hands" in status_text.lower():
                color = (0, 255, 0)
            elif "WAIT" in status_text:
                color = (0, 255, 255)
            elif "WARNING" in status_text:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    elif person_verified and not garbage_boxes:
        cv2.putText(display_frame, "Area Clean", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # --------------------------------------------------
    # 6. STATUS INDICATORS
    # --------------------------------------------------
    h, w = display_frame.shape[:2]
    
    if person_verified:
        cv2.circle(display_frame, (w - 25, 25), 8, (0, 255, 0), -1)
        cv2.putText(display_frame, "Person OK", (w - 110, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps = frame_count / elapsed
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # --------------------------------------------------
    # 7. EVIDENCE CAPTURE
    # --------------------------------------------------
    if littering_confirmed and not evidence_manager.is_recording:
        should_capture, reason, valid = evidence_manager.should_trigger_evidence(
            garbage_detections, person_detections, dustbin_boxes, littering_confirmed
        )
        
        if should_capture and valid:
            print("🚨 LITTERING CONFIRMED - Capturing evidence!")
            evidence_manager.start_evidence_capture(
                frame, valid, person_boxes, dustbin_boxes
            )
            littering_tracker.reset()
    
    # Recording indicator
    if evidence_manager.is_recording:
        evidence_manager.process_frame(frame)
        display_frame = evidence_manager.draw_recording_indicator(display_frame)
    
    # Cooldown
    cooldown = evidence_manager.get_cooldown_status()
    if cooldown:
        cv2.putText(display_frame, f"Cooldown: {cooldown}s", (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
    
    # --------------------------------------------------
    # 8. DISPLAY
    # --------------------------------------------------
    cv2.imshow("Smart Littering Detection", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("🔄 Manual reset")
        littering_tracker.reset()
        reset_garbage()
        reset_dustbin()
        reset_hand()
        person_frames = 0
        person_verified = False

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("\n✅ System stopped.")
