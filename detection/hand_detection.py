"""
Hand Detection Module — MediaPipe HandLandmarker
=================================================
Uses MediaPipe hand landmarks for precise hand tracking.
Detects 21 landmarks per hand — knows exactly where each finger is,
whether hand is open/closed/gripping.

Same API as the old YOLO-based module — drop-in replacement.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "hand_landmarker.task")

_hand_landmarker = None
_model_loaded = False


def _load_model():
    """Load MediaPipe HandLandmarker once."""
    global _hand_landmarker, _model_loaded
    if not _model_loaded:
        if os.path.exists(HAND_MODEL_PATH):
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                num_hands=4,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
            )
            _hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            print(f"✅ MediaPipe HandLandmarker loaded: {HAND_MODEL_PATH}")
        else:
            print(f"⚠️ Hand model not found at: {HAND_MODEL_PATH}")
            print("   Using person body fallback for hand regions")
            _hand_landmarker = None
        _model_loaded = True
    return _hand_landmarker


def detect_hands(frame, confidence_threshold=0.35, person_boxes=None):
    """
    Detect hands in frame using MediaPipe.

    Returns:
        hand_boxes: List of (x1, y1, x2, y2) bounding box tuples
    """
    landmarker = _load_model()

    if landmarker is None:
        if person_boxes:
            return _estimate_hand_regions_from_person(person_boxes)
        return []

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    hand_boxes = []
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1 = max(0, int(min(xs)) - 15)
            y1 = max(0, int(min(ys)) - 15)
            x2 = min(w, int(max(xs)) + 15)
            y2 = min(h, int(max(ys)) + 15)
            hand_boxes.append((x1, y1, x2, y2))

    # Filter by person proximity
    if person_boxes and hand_boxes:
        hand_boxes = _filter_hands_near_person(hand_boxes, person_boxes)

    return hand_boxes


def _estimate_hand_regions_from_person(person_boxes):
    """Fallback: estimate hand regions from person bounding boxes."""
    hand_regions = []
    for px1, py1, px2, py2 in person_boxes:
        pw = px2 - px1
        ph = py2 - py1
        expand_x = int(pw * 0.20)
        expand_y = int(ph * 0.05)
        hx1 = px1 - expand_x
        hx2 = px2 + expand_x
        hy1 = py1 + int(ph * 0.10)
        hy2 = py2 + expand_y
        hand_regions.append((hx1, hy1, hx2, hy2))
    return hand_regions


def _filter_hands_near_person(hand_boxes, person_boxes, max_distance=100):
    """Keep only hands near a detected person."""
    filtered = []
    for hx1, hy1, hx2, hy2 in hand_boxes:
        hcx = (hx1 + hx2) // 2
        hcy = (hy1 + hy2) // 2
        for px1, py1, px2, py2 in person_boxes:
            if (px1 - max_distance <= hcx <= px2 + max_distance and
                py1 - max_distance <= hcy <= py2 + max_distance):
                filtered.append((hx1, hy1, hx2, hy2))
                break
    return filtered


def draw_hands(frame, hand_boxes):
    """Draw hand bounding boxes on frame."""
    for x1, y1, x2, y2 in hand_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Hand", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame


def is_model_available():
    """Check if MediaPipe hand model exists."""
    return os.path.exists(HAND_MODEL_PATH)


def reset_tracker():
    """Reset (no-op for MediaPipe — stateless per-frame detection)."""
    pass
