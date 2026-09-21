import cv2
from ultralytics import YOLO
import os

# ============================================================
# DUSTBIN DETECTION - Using Trained YOLOv8 Model
# Only shows dustbin when ACTUALLY detected with high confidence
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "dustbin_detect.pt")

_dustbin_model = None
_model_loaded = False


def _load_model():
    """Load dustbin model once."""
    global _dustbin_model, _model_loaded
    if not _model_loaded:
        if os.path.exists(MODEL_PATH):
            _dustbin_model = YOLO(MODEL_PATH)
            print(f"✅ Dustbin model loaded: {MODEL_PATH}")
        else:
            print(f"❌ Dustbin model not found: {MODEL_PATH}")
        _model_loaded = True
    return _dustbin_model


# Stability tracker - only show after 5+ consecutive frames
class DustbinStabilityTracker:
    def __init__(self):
        self.tracked = {}  # id -> {box, count, last_seen}
        self.next_id = 0
        self.frame_num = 0
        self.min_frames = 5  # Must see 5 frames to be stable (stricter)
        self.max_missing = 3  # Remove after 3 missing frames (stricter)
    
    def update(self, raw_boxes):
        """Update with raw detections, return only stable ones."""
        self.frame_num += 1
        stable = []
        used_ids = set()
        
        for box in raw_boxes:
            x1, y1, x2, y2, conf = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            match_id = None
            min_dist = 100
            
            for tid, data in self.tracked.items():
                if tid in used_ids:
                    continue
                tx1, ty1, tx2, ty2, _ = data['box']
                tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    match_id = tid
            
            if match_id is not None:
                self.tracked[match_id]['box'] = box
                self.tracked[match_id]['count'] += 1
                self.tracked[match_id]['last_seen'] = self.frame_num
                used_ids.add(match_id)
                
                if self.tracked[match_id]['count'] >= self.min_frames:
                    stable.append(box)
            else:
                self.tracked[self.next_id] = {
                    'box': box,
                    'count': 1,
                    'last_seen': self.frame_num
                }
                self.next_id += 1
        
        to_del = [tid for tid, d in self.tracked.items() 
                  if self.frame_num - d['last_seen'] > self.max_missing]
        for tid in to_del:
            del self.tracked[tid]
        
        return stable
    
    def reset(self):
        self.tracked = {}


_tracker = DustbinStabilityTracker()


def detect_dustbin(frame, confidence_threshold=0.95):
    """
    Detect dustbins using trained model.
    Returns: (frame, list of dustbin boxes)
    Only returns boxes after stable detection (5+ frames)
    EXTREMELY HIGH confidence (95%) - Model needs retraining!
    Currently disabled due to false positives
    """
    model = _load_model()
    
    # TEMPORARILY DISABLED - Model detecting phone as dustbin!
    # Need to retrain with better dataset
    return frame, []
    
    if model is None:
        return frame, []
    
    results = model(frame, verbose=False, conf=confidence_threshold)
    
    raw_boxes = []
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                if w < 60 or h < 60 or area < 4000:
                    continue
                if area > (frame_area * 0.50):
                    continue
                if area < 2000 or area > 200000:
                    continue
                
                raw_boxes.append((x1, y1, x2, y2, conf))
    
    stable_boxes = _tracker.update(raw_boxes)
    
    dustbin_boxes = []
    for box in stable_boxes:
        x1, y1, x2, y2, conf = box
        dustbin_boxes.append((x1, y1, x2, y2))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        label = f"Dustbin {conf:.0%}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return frame, dustbin_boxes


def reset_tracker():
    """Reset the tracker."""
    _tracker.reset()
