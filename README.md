# DriveGuard AI 🚗💤

**Real-Time Driver Drowsiness & Fatigue Vision Alert System**

DriveGuard AI is a computer vision-driven driver safety system. It processes live browser webcam streams locally, computes 468 3D facial metrics (EAR, MAR, PERCLOS, Head Pose), analyzes temporal sequences using a PyTorch GRU neural network, triggers instant Web Audio safety alarms during microsleep or excessive yawning, and displays real-time telemetry on an interactive React dashboard.

---

## 🏗️ System Architecture

```
Browser Webcam (<video>)
      │
      ▼ (Canvas JPEG Frames ~15 FPS)
React 19 Dashboard (frontend/)
      │
      ▼ (WebSocket Stream: ws://localhost:8000/ws/telemetry)
FastAPI Backend (backend/api/main.py)
      │
      ▼
OpenCV Image Decoding (cv2.imdecode)
      │
      ▼
MediaPipe 3D FaceLandmarker (models/face_landmarker.task)
      │
      ├── Eye Aspect Ratio (EAR) & Microsleep State Machine
      ├── PERCLOS Rolling Window Calculator (% Closed Time)
      ├── Mouth Aspect Ratio (MAR) & Yawn Detector
      ├── Head Pose Estimator & Detector (Pitch / Yaw / Roll)
      ├── PyTorch 2-Layer GRU Temporal Neural Network (models/driveguard_gru.pth)
      └── Composite Fatigue Fusion Engine (0-100 Driver Fatigue Score)
      │
      ▼ (Real-Time JSON Telemetry Payload)
React Dashboard HUD
      ├── Live Metrics Cards (EAR, MAR, Alertness, Drowsy Frames, Yawns)
      ├── Recharts Time-Series Graph (EAR vs. MAR @ 5Hz)
      ├── Driver Camera Preview & AI Model Status Badges
      ├── Trip Incident Telemetry Log & JSON Export
      └── Browser Web Audio Synthesizer Alarm (85dB Pulsing Tone)
```

---

## 🎯 Key Features

- **Local Camera Vision Ingestion**: Browser `<video>` element with HTML5 `getUserMedia()` streaming compressed JPEG frames over WebSocket at ~15 FPS.
- **MediaPipe 3D Landmark Tracking**: 468 facial mesh landmarks mapped in real-time.
- **Eye Aspect Ratio (EAR)**: Dual-eye closure calculation with microsleep detection state machine (`NORMAL`, `CLOSING`, `DROWSY`, `RECOVERY`).
- **PERCLOS Rolling Window**: Time-based calculation of eye closure percentage over configurable rolling windows (60s default).
- **Mouth Aspect Ratio (MAR) & Yawning**: Temporal yawn counter tracking open mouth duration and excessive yawning (>3 yawns / 2 mins).
- **Head Pose Estimation**: 3D PnP head pose solver computing pitch, yaw, and roll angles to detect sustained head droop or driver distraction.
- **PyTorch GRU Neural Network**: Sequence prediction model trained on temporal EAR/MAR streams.
- **Composite Fatigue Engine**: Fuses multiple vision indicators into a unified 0-100 Driver Fatigue Score and Level (`NORMAL`, `WARNING`, `HIGH_RISK`, `CRITICAL`).
- **Web Audio Alarm System**: Non-blocking Web Audio synthesizer emitting an 85dB warning horn upon microsleep or critical fatigue.
- **100% Privacy-Preserving**: All computer vision inference runs locally on your device. Zero cloud uploads or third-party camera streaming.

---

## 📁 Repository Structure

```
driveguard-ai/
├── app/                        # Core Python AI Engine Package
│   ├── camera.py               # OpenCV VideoStream wrapper
│   ├── config.py               # Landmark indices, thresholds, & system constants
│   ├── eye_closure_detector.py # Temporal eye closure & microsleep state machine
│   ├── eye_metrics.py          # Left/Right/Avg EAR computation
│   ├── fatigue_engine.py       # Composite 0-100 Driver Fatigue Index fusion engine
│   ├── face_landmarker.py      # MediaPipe FaceLandmarker Task interface
│   ├── gru_model.py            # PyTorch 2-layer GRU neural network definition
│   ├── gru_predictor.py        # Sequence predictor for temporal fatigue GRU
│   ├── head_pose.py            # Head Pitch/Yaw/Roll PnP orientation solver
│   ├── mouth_metrics.py        # MAR computation
│   ├── perclos.py              # Time-based PERCLOS rolling window calculator
│   ├── telemetry.py            # Session analytics & telemetry serializer
│   └── yawn_detector.py        # Yawn state machine & 2-minute rolling counter
├── backend/
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI backend with REST endpoints & WebSocket handler
├── frontend/                   # React 19 + Vite Frontend
│   ├── src/
│   │   ├── App.jsx             # Cyberpunk Dashboard HUD, Web Audio Synthesizer, & Recharts
│   │   ├── main.jsx            # React root mount
│   │   └── index.css           # Tailwind v4 import & custom styling
│   ├── package.json            # React, Vite, Lucide React, Recharts dependencies
│   └── vite.config.js          # Vite build config
├── models/
│   ├── driveguard_gru.pth      # PyTorch GRU trained model weights
│   └── face_landmarker.task    # MediaPipe 468 3D landmark model bundle
├── scripts/
│   ├── demo_live_pipeline.py   # OpenCV desktop GUI HUD application
│   ├── prepare_temporal_dataset.py # Synthetic dataset generator for GRU
│   └── train_driveguard_gru.py # PyTorch training pipeline for GRU model
├── tests/                      # Pytest unit & integration test suite (60 passing tests)
├── requirements.txt            # Python dependencies (OpenCV, PyTorch, MediaPipe, FastAPI)
└── README.md                   # Project documentation
```

---

## 🚀 How to Run the Application

### 1. Prerequisites

- **Python 3.9+** (Python 3.10 / 3.11 / 3.12 / 3.13 supported)
- **Node.js 18+** & `npm`
- Working Webcam & Audio Output

### 2. Backend Setup & Run

1. Activate your virtual environment and install dependencies:
   ```bash
   # On Windows PowerShell:
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```bash
   uvicorn backend.api.main:app --reload --port 8000
   ```
   The backend will start at `http://localhost:8000`. You can verify server status at `http://localhost:8000/health`.

### 3. Frontend Setup & Run

1. Navigate to the `frontend/` folder:
   ```bash
   cd frontend
   npm install
   ```

2. Start the Vite development server:
   ```bash
   npm run dev
   ```

3. Open your browser at `http://localhost:5173`.

---

## 🔒 Permissions & Privacy

1. **Webcam Permission**: When opening the web dashboard, grant browser camera permissions. The video feed is rendered into a local `<video>` element and processed in memory by the local FastAPI backend.
2. **Audio Permission**: Click the **`🔊 ENABLE ALARM`** button in the header bar once to unblock browser AudioContext playback.
3. **Privacy Assurance**: All frame processing, landmark tracking, and GRU inference occur strictly on your local machine (`localhost`). No video data is recorded, saved, or uploaded to any remote servers.

---

## 🧪 Running Unit Tests

Run the full pytest suite from the project root:

```bash
$env:PYTHONPATH="."
.\.venv\Scripts\pytest.exe
```

---

## ❓ Troubleshooting

- **`ModuleNotFoundError: No module named 'app'`**:
  Ensure you start FastAPI using `uvicorn backend.api.main:app --reload --port 8000` from the project root directory.
- **Audio Alarm Silent**:
  Ensure you clicked the **`🔊 ENABLE ALARM`** button in the top-right header and your device volume is unmuted.
- **Camera Access Denied**:
  Check browser camera permissions in site settings (`chrome://settings/content/camera`).

---

## 🛡️ Model Information

- **MediaPipe FaceLandmarker**: Model bundle (`models/face_landmarker.task`) predicting 468 3D facial landmarks.
- **DriveGuard PyTorch GRU**: 2-layer recurrent neural network (`models/driveguard_gru.pth`) consuming sequence windows of `[ear_left, ear_right, ear_avg, mar]` to predict temporal driver fatigue states.
