"""
Camera Stream Ingestion Module for DriveGuard AI.

Handles video stream capture from local webcam or video files using OpenCV.
"""

from typing import Optional
import cv2
import numpy as np


class VideoStream:
    """Standard OpenCV Video Capture Stream wrapper."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def start(self) -> bool:
        """Initialize and start camera stream. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            self.is_running = False
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.is_running = True
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single BGR frame from the camera."""
        if not self.is_running or self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame

    def release(self) -> None:
        """Release camera hardware resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
