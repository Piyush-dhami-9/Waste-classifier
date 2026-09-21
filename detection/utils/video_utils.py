import cv2
from collections import deque


class FrameBuffer:
    """
    A buffer to store last N video frames.
    Uses deque for efficient add/remove.
    """
    
    def __init__(self, max_frames=150):
        self.frames = deque(maxlen=max_frames)
        self.max_frames = max_frames
    
    def add_frame(self, frame):
        """Add a new frame to the buffer."""
        self.frames.append(frame.copy())
    
    def get_frames(self):
        """Get all frames in the buffer."""
        return list(self.frames)
    
    def get_last_n_frames(self, n):
        """Get the last N frames from buffer."""
        frames_list = list(self.frames)
        return frames_list[-n:] if len(frames_list) >= n else frames_list
    
    def clear(self):
        """Clear all frames from buffer."""
        self.frames.clear()
    
    def is_full(self):
        """Check if buffer is full."""
        return len(self.frames) >= self.max_frames
    
    def __len__(self):
        """Return number of frames in buffer."""
        return len(self.frames)


def save_video(frames, filename, fps=30):
    """Save a list of frames as a video file."""
    if not frames:
        print("No frames to save!")
        return False
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved: {filename} ({len(frames)} frames)")
