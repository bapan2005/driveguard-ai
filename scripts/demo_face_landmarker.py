"""
Demo script to test MediaPipe Face Landmarker on a live webcam stream.

Press 'q' or ESC to exit the demo window.
"""

import sys
import os
import cv2

# Add parent directory to sys.path so app module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.face_landmarker import FaceLandmarkDetector


def main():
    print("==================================================")
    print(" DriveGuard AI - Face Landmarker Webcam Demo")
    print(" Press 'q' or ESC to quit.")
    print("==================================================")

    # Initialize Face Landmark Detector
    try:
        detector = FaceLandmarkDetector(auto_download=True)
    except Exception as e:
        print(f"[Error] Failed to initialize FaceLandmarkDetector: {e}")
        return

    # Open default webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam (device index 0). Please check your camera permissions.")
        return

    fps = 0
    frame_count = 0
    import time
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to read frame from webcam.")
            break

        frame_count += 1

        # Process frame with FaceLandmarkDetector
        result = detector.process_frame(frame)

        # Draw detected landmarks on frame
        if result is not None:
            annotated_frame = detector.draw_landmarks(frame, result, color=(0, 255, 0), radius=1)
            num_landmarks = len(result.pixel_landmarks)
            cv2.putText(
                annotated_frame,
                f"Face Detected | Landmarks: {num_landmarks}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            annotated_frame = frame.copy()
            cv2.putText(
                annotated_frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Calculate FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("DriveGuard AI - Face Landmarker Demo", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC key
            print("Exiting demo...")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
