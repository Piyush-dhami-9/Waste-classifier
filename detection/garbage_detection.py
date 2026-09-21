import cv2
from ultralytics import YOLO
import os
import time

# ============================================================
# GARBAGE DETECTION - Custom garbage_detect.pt Model
# NO FLICKERING - Uses state machine with hysteresis
# Uses person proximity for hold/drop detection
# Hand detection as optional accuracy enhancement
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "garbage_detect.pt")

_garbage_model = None
_model_loaded = False
_use_hand_detection = False  # Optional enhancement for hand-based tracking


def _load_model():
    """Load our custom garbage detection model."""
    global _garbage_model, _model_loaded
    if not _model_loaded:
        if os.path.exists(TRAINED_MODEL_PATH):
            _garbage_model = YOLO(TRAINED_MODEL_PATH)
            print(f"✅ Garbage model loaded: {TRAINED_MODEL_PATH}")
        else:
            print(f"❌ Garbage model not found: {TRAINED_MODEL_PATH}")
        
        _model_loaded = True
    return _garbage_model


class StableObjectTracker:
    """
    STABLE object tracker with state machine.
    
    States: NONE -> HOLDING -> DROPPED
    
    CRITICAL: No flickering - uses counters to confirm state changes.
    """
    
    def __init__(self):
        self.objects = {}  # id -> object state
        self.next_id = 0
        self.frame_num = 0
        
        # How many frames needed to confirm state change
        self.frames_to_confirm_hold = 10   # 10 frames (~0.3s) to confirm "holding"
        self.frames_to_confirm_drop = 12   # 12 frames (~0.4s) to confirm "dropped"
        self.frames_to_remove = 90         # 90 frames (~3 sec) to remove lost object
        
        # CRITICAL: Once dropped, need MANY more frames to go back to holding
        self.frames_to_rehold = 30         # 30 frames (~1 sec) to pick up again
    
    def update(self, detections, person_boxes, hand_boxes=None):
        """Update tracking and return stable results."""
        self.frame_num += 1
        matched_ids = set()
        
        # Match detections to existing tracks
        for det in detections:
            box = det['box']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Check if in hands NOW
            if hand_boxes and _use_hand_detection:
                in_hands_now = self._check_in_hands_with_hand_detection(box, hand_boxes, person_boxes)
            else:
                in_hands_now = self._check_in_hands(box, person_boxes)
            
            # Find matching track
            track_id = self._find_match(box)
            
            if track_id is not None:
                # Update existing track
                obj = self.objects[track_id]
                obj['box'] = box
                obj['class_name'] = class_name
                obj['confidence'] = confidence
                obj['last_seen'] = self.frame_num
                
                # Update state counters
                if in_hands_now:
                    obj['hold_count'] += 1
                    obj['drop_count'] = 0
                    obj['total_hold_frames'] = obj.get('total_hold_frames', 0) + 1
                else:
                    obj['drop_count'] += 1
                    obj['hold_count'] = 0
                
                # State transitions with hysteresis
                if obj['state'] == 'HOLDING':
                    if obj['drop_count'] >= self.frames_to_confirm_drop:
                        obj['state'] = 'DROPPED'
                        obj['drop_time'] = time.time()
                        print(f"🔴 Object DROPPED!")
                
                elif obj['state'] == 'DROPPED':
                    if obj['hold_count'] >= self.frames_to_rehold:
                        obj['state'] = 'HOLDING'
                        print(f"✅ Object picked up!")
                    elif obj['hold_count'] < 3:
                        obj['drop_count'] = max(obj['drop_count'], self.frames_to_confirm_drop)
                
                else:  # NONE state
                    if obj['hold_count'] >= self.frames_to_confirm_hold:
                        obj['state'] = 'HOLDING'
                    elif obj['drop_count'] >= self.frames_to_confirm_drop:
                        obj['state'] = 'DROPPED'
                        obj['drop_time'] = time.time()
                
                matched_ids.add(track_id)
            
            else:
                # New object - ONLY track if it starts IN HANDS
                if not in_hands_now:
                    continue
                
                new_id = self.next_id
                self.next_id += 1
                
                self.objects[new_id] = {
                    'box': box,
                    'class_name': class_name,
                    'confidence': confidence,
                    'state': 'NONE',
                    'hold_count': 1,
                    'drop_count': 0,
                    'total_hold_frames': 1,
                    'last_seen': self.frame_num,
                    'drop_time': None
                }
                matched_ids.add(new_id)
        
        # Handle objects not seen this frame
        to_remove = []
        for obj_id, obj in self.objects.items():
            if obj_id not in matched_ids:
                frames_missing = self.frame_num - obj['last_seen']
                
                if frames_missing > self.frames_to_remove:
                    to_remove.append(obj_id)
                else:
                    obj['drop_count'] += 1
                    obj['hold_count'] = 0
                    
                    if obj['state'] == 'HOLDING' and obj['drop_count'] >= self.frames_to_confirm_drop:
                        if obj.get('total_hold_frames', 0) >= 30:
                            obj['state'] = 'DROPPED'
                            obj['drop_time'] = time.time()
                            print(f"🔴 Object DROPPED (lost from view)!")
                        else:
                            to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.objects[obj_id]
        
        # Build results
        results = []
        for obj_id, obj in self.objects.items():
            if obj['state'] != 'NONE':
                results.append({
                    'box': obj['box'],
                    'class_name': obj['class_name'],
                    'confidence': obj['confidence'],
                    'in_hands': obj['state'] == 'HOLDING',
                    'state': obj['state'],
                    'drop_time': obj['drop_time']
                })
        
        return results
    
    def _check_in_hands(self, obj_box, person_boxes):
        """Object must be in person's hand/body region."""
        if not person_boxes:
            return False
        
        ox1, oy1, ox2, oy2 = obj_box
        obj_w = ox2 - ox1
        obj_h = oy2 - oy1
        obj_area = obj_w * obj_h
        
        if obj_area <= 0:
            return False
        
        for px1, py1, px2, py2 in person_boxes:
            pw = px2 - px1
            ph = py2 - py1
            
            expand_x = int(pw * 0.20)
            expand_y = int(ph * 0.05)
            
            hx1 = px1 - expand_x
            hx2 = px2 + expand_x
            hy1 = py1 + int(ph * 0.10)
            hy2 = py2 + expand_y
            
            ix1 = max(ox1, hx1)
            iy1 = max(oy1, hy1)
            ix2 = min(ox2, hx2)
            iy2 = min(oy2, hy2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if (intersection / obj_area) >= 0.30:
                    return True
        
        return False
    
    def _check_in_hands_with_hand_detection(self, obj_box, hand_boxes, person_boxes):
        """Use hand detection for better accuracy."""
        if not hand_boxes:
            return self._check_in_hands(obj_box, person_boxes)
        
        ox1, oy1, ox2, oy2 = obj_box
        obj_w = ox2 - ox1
        obj_h = oy2 - oy1
        obj_area = obj_w * obj_h
        
        if obj_area <= 0:
            return False
        
        for hx1, hy1, hx2, hy2 in hand_boxes:
            ix1 = max(ox1, hx1)
            iy1 = max(oy1, hy1)
            ix2 = min(ox2, hx2)
            iy2 = min(oy2, hy2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if (intersection / obj_area) >= 0.15:
                    return True
        
        return False
    
    def _find_match(self, box, threshold=100):
        """Find existing track matching this box."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        best_id = None
        best_dist = threshold
        
        for obj_id, obj in self.objects.items():
            ox1, oy1, ox2, oy2 = obj['box']
            ocx = (ox1 + ox2) // 2
            ocy = (oy1 + oy2) // 2
            
            dist = ((cx - ocx)**2 + (cy - ocy)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id
        
        return best_id
    
    def reset(self):
        """Reset all tracking."""
        self.objects = {}
        self.next_id = 0


_tracker = StableObjectTracker()


def get_garbage_detections(frame, confidence_threshold=0.20, person_boxes=None, hand_boxes=None):
    """
    Detect garbage with STABLE tracking.
    
    Returns list of detections with:
    - box, class_name, confidence
    - in_hands: True if HOLDING, False if DROPPED
    - state: 'HOLDING' or 'DROPPED'
    """
    if not person_boxes:
        _tracker.reset()
        return []
    
    model = _load_model()
    if model is None:
        return []
    
    raw_detections = []
    detected_boxes = []
    
    # Trained model
    results = model(frame, verbose=False, conf=confidence_threshold)
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id] if hasattr(model, 'names') else "Garbage"
                
                obj_box = (x1, y1, x2, y2)
                
                # Let ALL trained model detections through
                # The StableObjectTracker handles hold/drop via person proximity
                raw_detections.append({
                    'box': obj_box,
                    'class_name': class_name,
                    'confidence': conf
                })
                detected_boxes.append(obj_box)
    
    # Apply stable tracking
    stable_results = _tracker.update(raw_detections, person_boxes, hand_boxes)
    
    return stable_results


def reset_tracker():
    """Reset tracker."""
    _tracker.reset()


def enable_hand_detection(enabled=True):
    """Enable or disable hand detection mode."""
    global _use_hand_detection
    _use_hand_detection = enabled
    if enabled:
        print("✅ Hand detection enabled for garbage tracking")
    else:
        print("⚠️ Hand detection disabled, using person detection")
