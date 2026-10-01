"""
MediaPipe Face Landmarker Module for DriveGuard AI.

Uses the MediaPipe Tasks Face Landmarker API to detect 3D facial landmarks from OpenCV BGR frames.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = os.path.join("models", "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass
class LandmarkResult:
    """Dataclass holding extracted facial landmark coordinates for a single face."""
    normalized_landmarks: np.ndarray  # Shape: (N, 3) with columns [x, y, z] (normalized 0.0 - 1.0)
    pixel_landmarks: np.ndarray       # Shape: (N, 2) with columns [x, y] (pixel coordinates)


class FaceLandmarkDetector:
    """MediaPipe Tasks Face Landmarker wrapper for real-time single-face landmark extraction."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        auto_download: bool = True
    ):
        """
        Initialize the FaceLandmarkDetector.

        Args:
            model_path: Path to the local `face_landmarker.task` file.
            num_faces: Maximum number of faces to detect (default: 1 for driver monitoring).
            min_face_detection_confidence: Minimum confidence score for face detection.
            min_face_presence_confidence: Minimum confidence score for face presence.
            min_tracking_confidence: Minimum confidence score for landmark tracking.
            auto_download: Automatically download the model asset if missing.
        """
        self.model_path = os.path.abspath(model_path)
        self.num_faces = num_faces

        self._ensure_model_exists(auto_download=auto_download)
        self._init_landmarker(
            min_face_detection_confidence,
            min_face_presence_confidence,
            min_tracking_confidence
        )

    def _ensure_model_exists(self, auto_download: bool = True) -> None:
        """Verify that the model file exists; download if allowed and missing."""
        if os.path.exists(self.model_path):
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if auto_download:
            print(f"[FaceLandmarkDetector] Model file not found at '{self.model_path}'. Downloading from official CDN...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print(f"[FaceLandmarkDetector] Download completed successfully: '{self.model_path}'")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to automatically download MediaPipe Face Landmarker model from {MODEL_URL}. "
                    f"Error: {e}. Please manually download 'face_landmarker.task' and place it at '{self.model_path}'."
                ) from e
        else:
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model not found at '{self.model_path}'. "
                f"Please download 'face_landmarker.task' from {MODEL_URL} and place it in the models directory."
            )

    def _init_landmarker(
        self,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float
    ) -> None:
        """Initialize the MediaPipe Tasks FaceLandmarker object."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[LandmarkResult]:
        """
        Process an OpenCV BGR frame and extract facial landmarks for the driver's face.

        Args:
            frame_bgr: OpenCV image frame as a NumPy array (BGR format).

        Returns:
            LandmarkResult containing normalized (N, 3) and pixel (N, 2) coordinates,
            or None if no face was detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        # Extract driver's face (first detected face)
        first_face = detection_result.face_landmarks[0]

        normalized_coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in first_face],
            dtype=np.float32
        )

        pixel_coords = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in first_face],
            dtype=np.int32
        )

        return LandmarkResult(
            normalized_landmarks=normalized_coords,
            pixel_landmarks=pixel_coords
        )

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmark_result: LandmarkResult,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 1
    ) -> np.ndarray:
        """
        Render detected landmark points onto an OpenCV BGR frame.

        Args:
            frame_bgr: Source OpenCV image frame (BGR format).
            landmark_result: LandmarkResult object containing pixel coordinates.
            color: Circle color in BGR format (default: green).
            radius: Circle radius in pixels.

        Returns:
            Modified BGR frame with landmark overlay.
        """
        annotated_frame = frame_bgr.copy()
        if landmark_result is None or landmark_result.pixel_landmarks is None:
            return annotated_frame

        for x, y in landmark_result.pixel_landmarks:
            cv2.circle(annotated_frame, (x, y), radius, color, -1)

        return annotated_frame

    def close(self) -> None:
        """Release landmarker resources."""
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
