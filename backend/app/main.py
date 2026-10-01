import os
import sys
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from the DriveGuard project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.face_landmarker import FaceLandmarkDetector
from app.eye_metrics import EyeMetricsCalculator, EARMetrics
from app.eye_closure_detector import EyeClosureDetector, EyeClosureState
from app.mouth_metrics import MouthMetricsCalculator, MARMetrics
from app.yawn_detector import YawnDetector
from app.perclos import PERCLOSCalculator
from app.head_pose import HeadPoseEstimator, HeadPoseDetector, HeadPoseState, HeadPoseResult
from app.fatigue_engine import FatigueEngine, FatigueInput, FatigueLevel
from app.gru_predictor import GRUPredictor

app = FastAPI(title="DriveGuard AI Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def landmarks_to_numpy(normalized_landmarks):
    if normalized_landmarks is None:
        return None

    if (
        isinstance(normalized_landmarks, list)
        and normalized_landmarks
        and isinstance(normalized_landmarks[0], list)
    ):
        landmarks = normalized_landmarks[0]
    else:
        landmarks = normalized_landmarks

    try:
        return np.array(
            [[float(point.x), float(point.y)] for point in landmarks],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError):
        return None


def level_name(level):
    value = getattr(level, "value", level)
    return str(value).split(".")[-1].upper()


class VisionPipeline:
    """One stateful AI pipeline per browser/WebSocket session."""

    def __init__(self):
        self.landmarker = FaceLandmarkDetector(auto_download=True)
        self.eye_calculator = EyeMetricsCalculator()
        self.eye_detector = EyeClosureDetector(fps=20.0)
        self.mouth_calculator = MouthMetricsCalculator()
        self.yawn_detector = YawnDetector(fps=20.0)
        self.perclos_calculator = PERCLOSCalculator(fps=20.0)
        self.head_pose_estimator = HeadPoseEstimator(frame_size=(640, 480))
        self.head_pose_detector = HeadPoseDetector(fps=20.0)
        self.fatigue_engine = FatigueEngine()
        self.gru_predictor = GRUPredictor(
            model_path=os.path.join(PROJECT_ROOT, "models", "driveguard_gru.pth")
        )
        self.previous_time = time.perf_counter()
        self.consecutive_low_ear_frames = 0
        self.last_gru_result = None

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def process(self, frame):
        now = time.perf_counter()
        dt = now - self.previous_time
        self.previous_time = now
        if dt <= 0 or dt > 0.5:
            dt = 1.0 / 20.0

        landmark_result = self.landmarker.process_frame(frame)
        face_detected = (
            landmark_result is not None
            and landmark_result.normalized_landmarks is not None
        )

        if face_detected:
            landmarks = landmark_result.normalized_landmarks

            # EAR + microsleep state machine
            ear_metrics = self.eye_calculator.compute_metrics(landmarks)
            avg_ear = float(getattr(ear_metrics, "avg_ear", 0.0))
            eye_result = self.eye_detector.process_ear(
                avg_ear=avg_ear,
                is_valid=ear_metrics.is_valid,
            )

            if ear_metrics.is_valid and avg_ear < 0.22:
                self.consecutive_low_ear_frames += 1
            else:
                self.consecutive_low_ear_frames = 0

            # PERCLOS
            perclos_result = self.perclos_calculator.update(
                avg_ear=avg_ear,
                is_valid=ear_metrics.is_valid,
                dt=dt,
            )

            # MAR + yawn detector
            mar_value = float(self.mouth_calculator.calculate_mar(landmarks))
            mar_metrics = MARMetrics(
                mar=mar_value,
                mouth_open=mar_value >= 0.5,
                is_valid=True,
            )
            yawn_result = self.yawn_detector.process_mar(
                mar=mar_value,
                is_valid=True,
            )

            # GRU temporal model
            self.last_gru_result = self.gru_predictor.update(
                ear_left=float(ear_metrics.left_ear),
                ear_right=float(ear_metrics.right_ear),
                ear_avg=avg_ear,
                mar=mar_value,
            )

            # Head pose
            landmarks_np = landmarks_to_numpy(landmarks)
            if landmarks_np is not None:
                try:
                    head_pose_result = self.head_pose_estimator.estimate_pose(
                        landmarks_np,
                        frame_size=(frame.shape[1], frame.shape[0]),
                    )
                except TypeError:
                    head_pose_result = self.head_pose_estimator.estimate_pose(landmarks_np)
            else:
                head_pose_result = None

            if head_pose_result is None:
                head_pose_result = HeadPoseResult(
                    pitch=0.0, yaw=0.0, roll=0.0, is_valid=False
                )

            head_state_result = self.head_pose_detector.process_pose(
                head_pose_result,
                dt=dt,
            )

        else:
            ear_metrics = EARMetrics(
                left_ear=0.0,
                right_ear=0.0,
                avg_ear=0.0,
                is_valid=False,
            )
            eye_result = self.eye_detector.process_ear(
                avg_ear=0.0,
                is_valid=False,
            )
            perclos_result = self.perclos_calculator.update(
                avg_ear=0.0,
                is_valid=False,
                dt=dt,
            )
            mar_metrics = MARMetrics(
                mar=0.0,
                mouth_open=False,
                is_valid=False,
            )
            yawn_result = self.yawn_detector.process_mar(
                mar=0.0,
                is_valid=False,
            )
            head_pose_result = HeadPoseResult(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                is_valid=False,
            )
            head_state_result = self.head_pose_detector.process_pose(
                head_pose_result,
                dt=dt,
            )
            self.consecutive_low_ear_frames = 0

        fatigue_input = FatigueInput(
            microsleep_detected=bool(getattr(eye_result, "is_microsleep", False)),
            perclos_percent=float(getattr(perclos_result, "perclos_percentage", 0.0)),
            perclos_available=bool(getattr(perclos_result, "has_enough_data", False)),
            yawn_count=int(getattr(yawn_result, "yawns_in_window", 0)),
            excessive_yawning=bool(getattr(yawn_result, "excessive_yawning", False)),
            sustained_downward_pose=bool(getattr(head_state_result, "sustained_downward_pose", False)),
            head_looking_left=(getattr(head_state_result, "state", None) == HeadPoseState.LOOKING_LEFT),
            head_looking_right=(getattr(head_state_result, "state", None) == HeadPoseState.LOOKING_RIGHT),
        )
        fatigue_result = self.fatigue_engine.evaluate(fatigue_input)

        fatigue_score = float(getattr(fatigue_result, "score", 0.0))
        fatigue_level = level_name(getattr(fatigue_result, "level", FatigueLevel.NORMAL))
        alertness = max(0.0, min(100.0, 100.0 - fatigue_score))
        microsleep = bool(getattr(eye_result, "is_microsleep", False))
        yawn_warning = int(getattr(yawn_result, "yawns_in_window", 0)) >= 3

        gru_class = "COLLECTING"
        gru_confidence = 0.0
        if isinstance(self.last_gru_result, dict):
            gru_class = str(self.last_gru_result.get("class_name", "COLLECTING")).upper()
            gru_confidence = float(self.last_gru_result.get("confidence", 0.0))

        eye_state_str = getattr(eye_result, "state", EyeClosureState.NORMAL)
        if hasattr(eye_state_str, "value"):
            eye_state_str = eye_state_str.value

        return {
            "type": "telemetry",
            "face_detected": face_detected,
            "ear": round(float(getattr(ear_metrics, "avg_ear", 0.0)), 3),
            "ear_left": round(float(getattr(ear_metrics, "left_ear", 0.0)), 3),
            "ear_right": round(float(getattr(ear_metrics, "right_ear", 0.0)), 3),
            "mar": round(float(getattr(mar_metrics, "mar", 0.0)), 3),
            "eye_state": str(eye_state_str),
            "closed_eye_duration": round(float(getattr(eye_result, "closed_duration_seconds", 0.0)), 2),
            "consecutive_frames": int(self.consecutive_low_ear_frames),
            "microsleep": microsleep,
            "yawn_warning": yawn_warning,
            "yawn_count": int(getattr(yawn_result, "yawns_in_window", 0)),
            "perclos": round(float(getattr(perclos_result, "perclos_percentage", 0.0)), 2),
            "head_pitch": round(float(getattr(head_pose_result, "pitch", 0.0)), 1),
            "head_yaw": round(float(getattr(head_pose_result, "yaw", 0.0)), 1),
            "head_roll": round(float(getattr(head_pose_result, "roll", 0.0)), 1),
            "fatigue_score": round(fatigue_score, 2),
            "fatigue_level": fatigue_level,
            "gru_class": gru_class,
            "gru_confidence": round(gru_confidence, 4),
            "alertness": round(alertness, 1),
            "connection_status": "CONNECTED",
        }


@app.get("/")
async def root():
    return {
        "service": "DriveGuard AI Backend",
        "status": "running",
        "ai": "MediaPipe + EAR/MAR + PERCLOS + GRU + Fatigue Engine",
    }


@app.get("/health")
async def health():
    model_path = os.path.join(PROJECT_ROOT, "models", "driveguard_gru.pth")
    face_model_path = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")
    return {
        "status": "ok",
        "service": "DriveGuard AI Backend",
        "gru_model": os.path.exists(model_path),
        "face_landmarker": os.path.exists(face_model_path),
    }


@app.post("/api/sos/events")
async def record_sos_event(event: dict):
    """Record a demo emergency SOS escalation incident."""
    return {
        "status": "success",
        "message": "Emergency SOS event recorded",
        "event_id": event.get("eventId"),
        "timestamp": event.get("timestamp"),
    }


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    await websocket.accept()
    pipeline = None
    try:
        pipeline = VisionPipeline()
        await websocket.send_json({
            "type": "status",
            "backend_ready": True,
            "model_ready": True,
        })

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            frame_bytes = message.get("bytes")
            if not frame_bytes:
                continue

            np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            telemetry = pipeline.process(frame)
            await websocket.send_json(telemetry)

    except WebSocketDisconnect:
        print("Frontend disconnected")
    except Exception as exc:
        print(f"WebSocket AI pipeline error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if pipeline is not None:
            pipeline.close()
