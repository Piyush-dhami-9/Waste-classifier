# ============================================================
# EVIDENCE MANAGER - Smart Evidence Capture System
# Records ONLY confirmed littering events
# Saves: Screenshots, Videos, Event Logs
# ============================================================

import cv2
import os
import csv
import time
from datetime import datetime
from collections import deque


class EvidenceManager:
    """
    Manages evidence capture for confirmed littering events.
    
    Features:
    1. Screenshot capture with annotations
    2. Video recording (pre + post event)
    3. Confidence threshold filtering
    4. Distance-based validation
    5. Event logging (CSV)
    6. Cooldown timer (anti-duplicate)
    """
    
    def __init__(self, 
                 base_folder=None,
                 fps=30,
                 pre_buffer_seconds=5,
                 post_record_seconds=5,
                 cooldown_seconds=30,
                 min_garbage_confidence=0.50,
                 min_person_confidence=0.50):
        if base_folder is None:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_folder = os.path.join(_project_root, "evidence")
        
        self.base_folder = base_folder
        self.fps = fps
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_record_seconds = post_record_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_garbage_confidence = min_garbage_confidence
        self.min_person_confidence = min_person_confidence
        
        max_buffer_frames = fps * pre_buffer_seconds
        self.frame_buffer = deque(maxlen=max_buffer_frames)
        
        self.is_recording = False
        self.post_event_frames = []
        self.post_frame_count = 0
        self.event_data = None
        
        self.last_event_time = 0
        
        self._create_folders()
    
    def _create_folders(self):
        """Create evidence folder structure."""
        self.images_folder = os.path.join(self.base_folder, "images")
        self.videos_folder = os.path.join(self.base_folder, "videos")
        self.logs_folder = os.path.join(self.base_folder, "logs")
        
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.videos_folder, exist_ok=True)
        os.makedirs(self.logs_folder, exist_ok=True)
        
        self.log_file = os.path.join(self.logs_folder, "events.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'event_type', 'image_path', 'video_path', 
                               'garbage_class', 'garbage_confidence', 'person_confidence'])
    
    def add_frame(self, frame):
        """Add frame to rolling buffer."""
        self.frame_buffer.append(frame.copy())
    
    def is_in_cooldown(self):
        """Check if system is in cooldown period."""
        return (time.time() - self.last_event_time) < self.cooldown_seconds
    
    def check_confidence_threshold(self, garbage_detections, person_detections):
        """Filter detections by confidence threshold."""
        filtered_garbage = [
            d for d in garbage_detections 
            if d.get('confidence', 0) >= self.min_garbage_confidence
        ]
        
        filtered_persons = [
            p for p in person_detections 
            if p.get('confidence', 0) >= self.min_person_confidence
        ]
        
        passed = len(filtered_garbage) > 0
        
        return filtered_garbage, filtered_persons, passed
    
    def check_distance_from_dustbin(self, garbage_box, dustbin_boxes, min_distance_ratio=0.5):
        """Check if garbage is clearly outside dustbin area."""
        if not dustbin_boxes:
            return True, float('inf')
        
        gx1, gy1, gx2, gy2 = garbage_box
        garbage_center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
        
        min_distance = float('inf')
        
        for dustbin_box in dustbin_boxes:
            dx1, dy1, dx2, dy2 = dustbin_box
            dustbin_center = ((dx1 + dx2) // 2, (dy1 + dy2) // 2)
            dustbin_width = dx2 - dx1
            
            distance = ((garbage_center[0] - dustbin_center[0]) ** 2 + 
                       (garbage_center[1] - dustbin_center[1]) ** 2) ** 0.5
            
            min_distance = min(min_distance, distance)
            
            threshold = dustbin_width * min_distance_ratio
            if distance <= threshold:
                return False, distance
        
        return True, min_distance
    
    def should_trigger_evidence(self, garbage_detections, person_detections, dustbin_boxes, littering_confirmed):
        """Check ALL conditions before triggering evidence capture."""
        if self.is_recording:
            return False, "Already recording", None
        
        if not littering_confirmed:
            return False, "Littering not confirmed", None
        
        if self.is_in_cooldown():
            remaining = int(self.cooldown_seconds - (time.time() - self.last_event_time))
            return False, f"Cooldown active ({remaining}s remaining)", None
        
        filtered_garbage, filtered_persons, passed = self.check_confidence_threshold(
            garbage_detections, person_detections
        )
        
        if not passed:
            return False, "Confidence threshold not met", None
        
        for garbage in filtered_garbage:
            is_outside, distance = self.check_distance_from_dustbin(
                garbage['box'], dustbin_boxes
            )
            
            if is_outside:
                return True, "All conditions met", garbage
        
        return False, "Garbage too close to dustbin", None
    
    def start_evidence_capture(self, frame, garbage_data, person_boxes, dustbin_boxes):
        """Start evidence capture process."""
        self.is_recording = True
        self.post_frame_count = 0
        
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.event_data = {
            'timestamp': timestamp,
            'timestamp_str': timestamp_str,
            'garbage': garbage_data,
            'person_boxes': person_boxes,
            'dustbin_boxes': dustbin_boxes,
            'image_path': None,
            'video_path': None
        }
        
        self.post_event_frames = list(self.frame_buffer)
        
        self._save_screenshot(frame)
        
        print("\n" + "=" * 60)
        print("🚨 LITTERING EVIDENCE CAPTURE STARTED")
        print("=" * 60)
        print(f"📅 Time: {timestamp_str}")
        print(f"🗑️ Object: {garbage_data.get('class_name', 'unknown')}")
        print(f"📊 Confidence: {garbage_data.get('confidence', 0):.2f}")
        print(f"📸 Pre-event frames: {len(self.post_event_frames)}")
        print(f"⏱️ Recording {self.post_record_seconds}s post-event...")
    
    def _save_screenshot(self, frame):
        """Save annotated screenshot."""
        annotated = frame.copy()
        
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), (0, 0, 150), -1)
        cv2.putText(annotated, "LITTERING DETECTED", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        timestamp_text = self.event_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp_text, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        garbage = self.event_data['garbage']
        if garbage:
            x1, y1, x2, y2 = garbage['box']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
            label = f"GARBAGE: {garbage.get('class_name', 'unknown')}"
            cv2.putText(annotated, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for px1, py1, px2, py2 in self.event_data.get('person_boxes', []):
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(annotated, "PERSON", (px1, py1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for dx1, dy1, dx2, dy2 in self.event_data.get('dustbin_boxes', []):
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
            cv2.putText(annotated, "DUSTBIN", (dx1, dy1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        filename = f"litter_{self.event_data['timestamp_str']}.jpg"
        filepath = os.path.join(self.images_folder, filename)
        cv2.imwrite(filepath, annotated)
        self.event_data['image_path'] = filepath
        
        print(f"📸 Screenshot saved: {filename}")
    
    def process_frame(self, frame):
        """Process frame during post-event recording."""
        if not self.is_recording:
            return False
        
        self.post_event_frames.append(frame.copy())
        self.post_frame_count += 1
        
        target_frames = self.fps * self.post_record_seconds
        
        if self.post_frame_count >= target_frames:
            self._finalize_evidence()
            return False
        
        return True
    
    def _finalize_evidence(self):
        """Save video and log event."""
        filename = f"litter_{self.event_data['timestamp_str']}.mp4"
        filepath = os.path.join(self.videos_folder, filename)
        
        self._save_video(self.post_event_frames, filepath)
        self.event_data['video_path'] = filepath
        
        self._log_event()
        
        self.last_event_time = time.time()
        
        self.is_recording = False
        self.post_event_frames = []
        self.post_frame_count = 0
        
        print(f"🎥 Video saved: {filename}")
        print(f"📝 Event logged to CSV")
        print(f"⏳ Cooldown: {self.cooldown_seconds}s before next capture")
        print("=" * 60)
        print("✅ EVIDENCE CAPTURE COMPLETE")
        print("=" * 60 + "\n")
    
    def _save_video(self, frames, filepath):
        """Save frames as video file."""
        if not frames:
            return False
        
        height, width = frames[0].shape[:2]
        
        codecs_to_try = [
            ('avc1', '.mp4'),
            ('H264', '.mp4'),
            ('X264', '.mp4'),
            ('XVID', '.avi'),
            ('mp4v', '.mp4'),
        ]
        
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                if ext != '.mp4':
                    filepath = filepath.replace('.mp4', ext)
                
                writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
                
                if writer.isOpened():
                    for frame in frames:
                        writer.write(frame)
                    writer.release()
                    return True
                else:
                    writer.release()
            except Exception:
                continue
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return True
    
    def _log_event(self):
        """Append event to CSV log."""
        garbage = self.event_data.get('garbage', {})
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.event_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'littering',
                self.event_data.get('image_path', ''),
                self.event_data.get('video_path', ''),
                garbage.get('class_name', 'unknown'),
                f"{garbage.get('confidence', 0):.2f}",
                ''
            ])
    
    def get_recording_status(self):
        """Get current recording status for display."""
        if not self.is_recording:
            return None
        
        progress = self.post_frame_count / (self.fps * self.post_record_seconds)
        return {
            'is_recording': True,
            'progress': progress,
            'frames_recorded': self.post_frame_count,
            'total_frames': self.fps * self.post_record_seconds
        }
    
    def draw_recording_indicator(self, frame):
        """Draw recording indicator on frame."""
        if not self.is_recording:
            return frame
        
        cv2.circle(frame, (50, 100), 15, (0, 0, 255), -1)
        
        cv2.putText(frame, "REC EVIDENCE", (75, 108), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        progress = self.post_frame_count / (self.fps * self.post_record_seconds)
        bar_width = 200
        bar_x = 50
        bar_y = 130
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (0, 0, 255), -1)
        
        return frame
    
    def get_cooldown_status(self):
        """Get cooldown status for display."""
        if not self.is_in_cooldown():
            return None
        
        remaining = int(self.cooldown_seconds - (time.time() - self.last_event_time))
        return remaining
