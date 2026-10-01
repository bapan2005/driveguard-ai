"""
DriveGuard AI - Live Vision Pipeline Demonstration

Pipeline:
    OpenCV Webcam
        ↓
    MediaPipe Face Landmarker
        ↓
    Facial Landmarks
        ├── EAR → Eye Closure / Microsleep
        ├── MAR → Yawn Detection
        └── Head Pose → Downward / Sideways Head
        ↓
    PERCLOS & GRU Temporal Model
        ↓
    Fatigue Engine
        ↓
    Live HUD + Risk Status

Press 'q' or ESC to quit.
"""

from app.gru_predictor import GRUPredictor
from app.perclos import PERCLOSCalculator
from app.alerts import AlarmManager
import sys
import os
import time

import cv2
import numpy as np

# Add project root to Python path.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from app.camera import VideoStream
from app.face_landmarker import FaceLandmarkDetector

from app.eye_metrics import (
    EyeMetricsCalculator,
    EARMetrics,
)
from app.eye_closure_detector import (
    EyeClosureDetector,
    EyeClosureState,
)

from app.mouth_metrics import (
    MouthMetricsCalculator,
    MARMetrics,
)
from app.yawn_detector import (
    YawnDetector,
    YawnState,
)

from app.head_pose import (
    HeadPoseEstimator,
    HeadPoseDetector,
    HeadPoseState,
    HeadPoseResult,
)

from app.fatigue_engine import (
    FatigueEngine,
    FatigueInput,
    FatigueLevel,
)

# ============================================================
# Helper: Convert MediaPipe landmarks to NumPy coordinates
# ============================================================

def landmarks_to_numpy(normalized_landmarks):
    """
    Convert MediaPipe normalized landmarks to a NumPy array.

    Returns:
        NumPy array with shape (N, 2), containing normalized
        x and y coordinates.
    """
    if normalized_landmarks is None:
        return None

    # Handle a possible list containing one face.
    if (
        isinstance(normalized_landmarks, list)
        and len(normalized_landmarks) > 0
        and isinstance(normalized_landmarks[0], list)
    ):
        landmarks = normalized_landmarks[0]
    else:
        landmarks = normalized_landmarks

    try:
        points = np.array(
            [
                [float(point.x), float(point.y)]
                for point in landmarks
            ],
            dtype=np.float64,
        )
        return points
    except (AttributeError, TypeError, ValueError):
        return None


# ============================================================
# HUD
# ============================================================

def draw_hud_overlay(
    frame: np.ndarray,
    fps: float,
    face_detected: bool,
    ear_metrics: EARMetrics,
    eye_result,
    mar_metrics: MARMetrics,
    yawn_result,
    perclos_result=None,
    head_pose_result=None,
    fatigue_result=None,
   gru_result=None, 
) -> np.ndarray:

    # Backward-compatible defaults for existing tests/callers.
    if perclos_result is None:
        class DefaultPERCLOSResult:
            perclos_percentage = 0.0
            total_valid_seconds = 0.0
            closed_seconds = 0.0
            has_enough_data = False

        perclos_result = DefaultPERCLOSResult()

    if fatigue_result is None:
        class DefaultFatigueResult:
            score = 0.0
            level = FatigueLevel.NORMAL
            reasons = []

        fatigue_result = DefaultFatigueResult()

    if head_pose_result is None:
        class DefaultHeadPoseResult:
            pitch = 0.0
            yaw = 0.0
            roll = 0.0
            is_valid = False

        head_pose_result = DefaultHeadPoseResult()

    hud_frame = frame.copy()
    h, w, _ = hud_frame.shape

    # --------------------------------------------------------
    # Colors - BGR
    # --------------------------------------------------------
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (150, 150, 150)
    COLOR_DARK = (20, 20, 20)

    # --------------------------------------------------------
    # Determine overall status
    # --------------------------------------------------------
    if not face_detected:
        status_text = "FACE NOT DETECTED"
        status_color = COLOR_GRAY
    elif fatigue_result.level == FatigueLevel.CRITICAL:
        status_text = "CRITICAL FATIGUE ALERT"
        status_color = COLOR_RED
    elif fatigue_result.level == FatigueLevel.HIGH_RISK:
        status_text = "HIGH FATIGUE RISK"
        status_color = COLOR_RED
    elif fatigue_result.level == FatigueLevel.WARNING:
        status_text = "FATIGUE WARNING"
        status_color = COLOR_ORANGE
    elif eye_result.state == EyeClosureState.CLOSING:
        status_text = "EYES CLOSING"
        status_color = COLOR_YELLOW
    elif yawn_result.state == YawnState.YAWNING:
        status_text = "YAWN DETECTED"
        status_color = COLOR_ORANGE
    else:
        status_text = "DRIVER ALERT / NORMAL"
        status_color = COLOR_GREEN

    # --------------------------------------------------------
    # Top header
    # --------------------------------------------------------
    overlay = hud_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 135), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.75, hud_frame, 0.25, 0, hud_frame)

    cv2.putText(
        hud_frame,
        f"STATUS: {status_text}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
    )

    face_text = "YES" if face_detected else "NO"
    cv2.putText(
        hud_frame,
        f"FPS: {fps:.1f} | Face: {face_text}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLOR_WHITE,
        1,
    )

    if face_detected:
        avg_ear = getattr(ear_metrics, 'avg_ear', getattr(ear_metrics, 'average_ear', 0.0))
        cv2.putText(
            hud_frame,
            f"EAR: {avg_ear:.3f} (L:{ear_metrics.left_ear:.2f} R:{ear_metrics.right_ear:.2f})",
            (15, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_CYAN,
            1,
        )
        cv2.putText(
            hud_frame,
            f"MAR: {mar_metrics.mar:.3f}",
            (15, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_CYAN,
            1,
        )

    if perclos_result.has_enough_data:
        perclos_text = f"PERCLOS: {perclos_result.perclos_percentage:.1f}%"
        if perclos_result.perclos_percentage >= 40:
            perclos_color = COLOR_RED
        elif perclos_result.perclos_percentage >= 20:
            perclos_color = COLOR_ORANGE
        else:
            perclos_color = COLOR_GREEN
    else:
        perclos_text = f"PERCLOS: collecting ({perclos_result.total_valid_seconds:.1f}s)"
        perclos_color = COLOR_GRAY

    cv2.putText(
        hud_frame,
        perclos_text,
        (15, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        perclos_color,
        1,
    )

    # --------------------------------------------------------
    # Right metrics card
    # --------------------------------------------------------
    card_width = 285
    card_height = 245
    card_x = max(10, w - card_width - 10)
    card_y = 10

    overlay_card = hud_frame.copy()
    cv2.rectangle(
        overlay_card,
        (card_x, card_y),
        (w - 10, card_y + card_height),
        COLOR_DARK,
        -1,
    )
    cv2.addWeighted(overlay_card, 0.78, hud_frame, 0.22, 0, hud_frame)

    cv2.putText(
        hud_frame,
        "DRIVER TELEMETRY",
        (card_x + 10, card_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLOR_WHITE,
        1,
    )

    cv2.putText(
        hud_frame,
        f"Eye State: {eye_result.state.value}",
        (card_x + 10, card_y + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (
            COLOR_RED
            if eye_result.is_microsleep
            else COLOR_YELLOW
            if eye_result.state == EyeClosureState.CLOSING
            else COLOR_WHITE
        ),
        1,
    )

    cv2.putText(
        hud_frame,
        f"Closed: {eye_result.closed_duration_seconds:.1f}s",
        (card_x + 10, card_y + 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        COLOR_WHITE,
        1,
    )

    cv2.putText(
        hud_frame,
        f"PERCLOS: {perclos_result.perclos_percentage:.1f}%",
        (card_x + 10, card_y + 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        perclos_color,
        1,
    )

    cv2.putText(
        hud_frame,
        f"Yawns Total: {yawn_result.total_yawns}",
        (card_x + 10, card_y + 113),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        COLOR_WHITE,
        1,
    )

    cv2.putText(
        hud_frame,
        f"Yawns / 2min: {yawn_result.yawns_in_window}",
        (card_x + 10, card_y + 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        COLOR_RED if yawn_result.excessive_yawning else COLOR_WHITE,
        1,
    )

    if head_pose_result.is_valid:
        cv2.putText(
            hud_frame,
            f"Pitch: {head_pose_result.pitch:.1f}",
            (card_x + 10, card_y + 157),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            COLOR_WHITE,
            1,
        )
        cv2.putText(
            hud_frame,
            f"Yaw: {head_pose_result.yaw:.1f}",
            (card_x + 10, card_y + 179),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            COLOR_WHITE,
            1,
        )
        cv2.putText(
            hud_frame,
            f"Roll: {head_pose_result.roll:.1f}",
            (card_x + 10, card_y + 201),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            COLOR_WHITE,
            1,
        )
    else:
        cv2.putText(
            hud_frame,
            "Head Pose: N/A",
            (card_x + 10, card_y + 179),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            COLOR_GRAY,
            1,
        )

    score_color = COLOR_GREEN
    if fatigue_result.level == FatigueLevel.WARNING:
        score_color = COLOR_ORANGE
    elif fatigue_result.level in (FatigueLevel.HIGH_RISK, FatigueLevel.CRITICAL):
        score_color = COLOR_RED

    cv2.putText(
        hud_frame,
        f"Fatigue Score: {fatigue_result.score:.1f}/100",
        (card_x + 10, card_y + 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        score_color,
        2,
    )
        # --------------------------------------------------------
    # GRU AI prediction
    # --------------------------------------------------------
    if gru_result is not None:
        gru_text = (
            f"AI Model: {gru_result['class_name']} "
            f"({gru_result['confidence'] * 100:.1f}%)"
        )

        cv2.putText(
            hud_frame,
            gru_text,
            (card_x + 10, card_y + 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_WHITE,
            1,
        )
    else:
        cv2.putText(
            hud_frame,
            "AI Model: Collecting...",
            (card_x + 10, card_y + 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_GRAY,
            1,
        )

    # --------------------------------------------------------
    # Bottom alert banners
    # --------------------------------------------------------
    if fatigue_result.level == FatigueLevel.CRITICAL:
        cv2.rectangle(hud_frame, (0, 0), (w - 1, h - 1), COLOR_RED, 8)
        banner_overlay = hud_frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 72), (w, h), COLOR_RED, -1)
        cv2.addWeighted(banner_overlay, 0.85, hud_frame, 0.15, 0, hud_frame)
        cv2.putText(
            hud_frame,
            "CRITICAL FATIGUE - TAKE A BREAK",
            (max(20, w // 2 - 220), h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            COLOR_WHITE,
            2,
        )
    elif fatigue_result.level == FatigueLevel.HIGH_RISK:
        banner_overlay = hud_frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 60), (w, h), COLOR_RED, -1)
        cv2.addWeighted(banner_overlay, 0.8, hud_frame, 0.2, 0, hud_frame)
        cv2.putText(
            hud_frame,
            "HIGH FATIGUE RISK",
            (max(20, w // 2 - 130), h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            COLOR_WHITE,
            2,
        )
    elif fatigue_result.level == FatigueLevel.WARNING:
        banner_overlay = hud_frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 52), (w, h), COLOR_ORANGE, -1)
        cv2.addWeighted(banner_overlay, 0.8, hud_frame, 0.2, 0, hud_frame)
        cv2.putText(
            hud_frame,
            "FATIGUE WARNING",
            (max(20, w // 2 - 120), h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_WHITE,
            2,
        )
    elif yawn_result.excessive_yawning:
        banner_overlay = hud_frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 45), (w, h), COLOR_ORANGE, -1)
        cv2.addWeighted(banner_overlay, 0.8, hud_frame, 0.2, 0, hud_frame)
        cv2.putText(
            hud_frame,
            "EXCESSIVE YAWNING DETECTED",
            (max(20, w // 2 - 160), h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            COLOR_WHITE,
            2,
        )

    return hud_frame


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 55)
    print(" DriveGuard AI - Live Vision Pipeline Demo")
    print("=" * 55)
    print(" Press 'q' or ESC to quit.")
    print("=" * 55)

    try:
        landmarker = FaceLandmarkDetector(auto_download=True)
    except Exception as e:
        print(f"[Error] Failed to initialize FaceLandmarkDetector: {e}")
        return

    # Initialize detectors
    eye_calculator = EyeMetricsCalculator()
    eye_detector = EyeClosureDetector(fps=30.0)
    mouth_calculator = MouthMetricsCalculator()
    yawn_detector = YawnDetector(fps=30.0)
    perclos_calculator = PERCLOSCalculator(fps=30.0)
    head_pose_estimator = HeadPoseEstimator(frame_size=(640, 480))
    head_pose_detector = HeadPoseDetector(fps=30.0)
    fatigue_engine = FatigueEngine()
    
    gru_predictor = GRUPredictor(model_path="models/driveguard_gru.pth")

    alarm_manager = AlarmManager(
        alarm_sound_path=os.path.join(
            os.path.dirname(__file__), "..", "assets", "alarm.wav"
        ),
        yawn_sound_path=os.path.join(
            os.path.dirname(__file__), "..", "assets", "yawn_warning.wav"
        )
    )

    camera = VideoStream(device_index=0, width=640, height=480)

    if not camera.start():
        print("[Error] Could not open webcam (device index 0).")
        landmarker.close()
        return

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    previous_time = start_time

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("[Warning] Camera feed disconnected or returned empty frame.")
                break

            frame_count += 1
            current_time = time.time()
            frame_dt = current_time - previous_time
            if frame_dt <= 0:
                frame_dt = 1.0 / 30.0
            previous_time = current_time

            elapsed = current_time - start_time
            if elapsed > 0:
                fps = frame_count / elapsed

            landmark_result = landmarker.process_frame(frame)
            face_detected = (
                landmark_result is not None
                and landmark_result.normalized_landmarks is not None
            )

            head_pose_result = None

            if face_detected:
                landmarks = landmark_result.normalized_landmarks

                # EAR
                ear_metrics = eye_calculator.compute_metrics(landmarks)
                avg_ear = getattr(ear_metrics, 'avg_ear', getattr(ear_metrics, 'average_ear', 0.0))

                eye_result = eye_detector.process_ear(
                    avg_ear=avg_ear,
                    is_valid=ear_metrics.is_valid,
                )

                # PERCLOS
                perclos_result = perclos_calculator.update(
                    avg_ear=avg_ear,
                    is_valid=ear_metrics.is_valid,
                    dt=frame_dt,
                )

                # MAR / YAWN
                mar_value = mouth_calculator.calculate_mar(landmarks)

                mar_metrics = MARMetrics(
                    mar=mar_value,
                    mouth_open=mar_value >= 0.5,
                    is_valid=True,
                )

                yawn_result = yawn_detector.process_mar(
                    mar=mar_value,
                    is_valid=True,
                )

                # GRU TEMPORAL PREDICTION
                gru_result = gru_predictor.update(
                    ear_left=ear_metrics.left_ear,
                    ear_right=ear_metrics.right_ear,
                    ear_avg=avg_ear,
                    mar=mar_value,
                )
                # HEAD POSE
                landmarks_np = landmarks_to_numpy(landmarks)

                if landmarks_np is not None:
                    try:
                        head_pose_result = head_pose_estimator.estimate_pose(
                            landmarks_np,
                            frame_size=(frame.shape[1], frame.shape[0]),
                        )
                    except TypeError:
                        head_pose_result = head_pose_estimator.estimate_pose(landmarks_np)

                    if head_pose_result is None:
                        head_pose_result = HeadPoseResult(
                            pitch=0.0, yaw=0.0, roll=0.0, is_valid=False
                        )
                else:
                    head_pose_result = HeadPoseResult(
                        pitch=0.0, yaw=0.0, roll=0.0, is_valid=False
                    )

                head_state_result = head_pose_detector.process_pose(
                    head_pose_result, dt=frame_dt
                )

                annotated_frame = landmarker.draw_landmarks(
                    frame,
                    landmark_result,
                    color=(0, 255, 0),
                    radius=1,
                )

            else:
                # No face
                ear_metrics = EARMetrics(
                    left_ear=0.0, right_ear=0.0, avg_ear=0.0, is_valid=False
                )
                eye_result = eye_detector.process_ear(avg_ear=0.0, is_valid=False)
                perclos_result = perclos_calculator.update(
                    avg_ear=0.0, is_valid=False, dt=frame_dt
                )
                mar_metrics = MARMetrics(
                    mar=0.0, mouth_open=False, is_valid=False
                )
                yawn_result = yawn_detector.process_mar(mar=0.0, is_valid=False)

                invalid_head_pose = HeadPoseResult(
                    pitch=0.0, yaw=0.0, roll=0.0, is_valid=False
                )
                head_state_result = head_pose_detector.process_pose(
                    invalid_head_pose, dt=frame_dt
                )
                annotated_frame = frame.copy()

            # Fatigue Engine
            fatigue_input = FatigueInput(
                microsleep_detected=eye_result.is_microsleep,
                perclos_percent=perclos_result.perclos_percentage,
                perclos_available=perclos_result.has_enough_data,
                yawn_count=yawn_result.yawns_in_window,
                excessive_yawning=yawn_result.excessive_yawning,
                sustained_downward_pose=head_state_result.sustained_downward_pose,
                head_looking_left=(
                    head_state_result.state == HeadPoseState.LOOKING_LEFT
                ),
                head_looking_right=(
                    head_state_result.state == HeadPoseState.LOOKING_RIGHT
                ),
            )

            fatigue_result = fatigue_engine.evaluate(fatigue_input)

            # Audio Alerts
            if yawn_result.yawns_in_window >= 3:
                alarm_manager.trigger_yawn_warning()

            if fatigue_result.level in (
                FatigueLevel.HIGH_RISK,
                FatigueLevel.CRITICAL,
            ):
                alarm_manager.trigger_alarm()
            else:
                alarm_manager.stop_alarm()

            # HUD Rendering
            hud_frame = draw_hud_overlay(
                frame=annotated_frame,
                fps=fps,
                face_detected=face_detected,
                ear_metrics=ear_metrics,
                eye_result=eye_result,
                mar_metrics=mar_metrics,
                yawn_result=yawn_result,
                perclos_result=perclos_result,
                head_pose_result=(
                    head_pose_result
                    if head_pose_result is not None
                    else head_state_result
                ),
                fatigue_result=fatigue_result,
                gru_result=gru_result,
            )

            cv2.imshow("DriveGuard AI - Live Vision Pipeline", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Exiting live pipeline demo...")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()