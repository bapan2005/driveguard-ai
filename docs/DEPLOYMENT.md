# DriveGuard AI - Deployment & Execution Guide

## 💻 Local System Setup

### 1. Environment Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.9, 3.10, or 3.11 recommended
- **Hardware**: Standard USB Webcam or integrated laptop webcam

### 2. Quickstart Commands

```bash
# 1. Clone workspace
git clone https://github.com/your-username/driveguard-ai.git
cd driveguard-ai

# 2. Setup Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # macOS/Linux

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Launch Streamlit Dashboard
streamlit run dashboard/app.py
```

### 3. Troubleshooting
- **Webcam permission issues**: Ensure browser / Python terminal has webcam privacy permissions granted in system settings.
- **Audio alarm silent**: Check speaker volume and ensure Pygame mixer or system sound drivers are active.
