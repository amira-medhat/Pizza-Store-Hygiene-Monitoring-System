# pizza_monitoring/streaming_service/state.py

import cv2
import threading

# Thread-safe latest frame storage
latest_frame = None
frame_lock = threading.Lock()

def update_frame(frame):
    """Thread-safe update of the latest frame"""
    global latest_frame
    with frame_lock:
        latest_frame = frame

def get_frame():
    """Thread-safe retrieval of the latest frame"""
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None