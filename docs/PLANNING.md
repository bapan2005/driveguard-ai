# DriveGuard AI - Project Planning & Milestones

## 📌 Project Objectives

1. **Live Video Processing**: Ingest webcam feed with $\ge 25\text{ FPS}$ on standard CPU.
2. **Landmark & Metric Computation**: Extract MediaPipe 3D face mesh landmarks, compute EAR, MAR, and Head Pose.
3. **Driver State Machine**: Track eye closures, microsleep events, yawn frequency, and calculate dynamic Fatigue Index (0 - 100%).
4. **Alert System**: Non-blocking audio playback and visual HUD warning borders.
5. **Dashboard & Telemetry**: Interactive Streamlit application with live video feed and dynamic Plotly timeline charts.

---

## 🚀 Execution Phases

### Phase 1: Environment & Modular Folder Structure Setup
- Set up directory structure, `.gitignore`, `requirements.txt`, and module stubs.
- Define central `app/config.py` hyperparameters.

### Phase 2: Core Computer Vision Modules
- Implement `app/face_landmarker.py` (MediaPipe Face Mesh integration).
- Implement `app/eye_metrics.py` (Euclidean EAR formula for left & right eyes).
- Implement `app/mouth_metrics.py` (MAR formula & lip opening).
- Implement `app/head_pose.py` (Head orientation via OpenCV `solvePnP`).

### Phase 3: State Machine & Alert Engine
- Implement `app/fatigue_engine.py` (Frame counters, state classifier, dynamic fatigue score).
- Implement `app/alerts.py` (Threaded non-blocking audio player & OpenCV HUD renderer).

### Phase 4: Telemetry & Dashboard Integration
- Implement `app/telemetry.py` (In-memory rolling history buffer).
- Implement `dashboard/app.py` (Streamlit web dashboard layout, controls, and Plotly graphs).

### Phase 5: Testing, Calibration & Hackathon Defense Prep
- Benchmark FPS and latency.
- Calibrate EAR/MAR sensitivity under varying lighting.
- Prepare hackathon pitch & defense QA guide (`docs/DEFENSE_QA.md`).
