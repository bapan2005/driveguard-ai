# DriveGuard AI - Hackathon Defense & Judge Q&A Guide

## 🎯 Frequently Asked Questions & Technical Defense Strategy

### Q1: Why use MediaPipe Face Mesh instead of Haar Cascades or Dlib?
**Answer**:
- **Dlib**: Heavy C++ dependency (dlib build compilation issues on Windows), slower CPU inference frame rates.
- **Haar Cascades**: Highly sensitive to lighting variations, frequent false positives/negatives, lacks 3D landmark depth.
- **MediaPipe Face Mesh**: Provides 468 high-precision 3D facial landmarks out of the box, optimized for real-time mobile and edge CPU execution ($\ge 30\text{ FPS}$), highly robust to head rotation and illumination changes.

---

### Q2: How do you prevent audio alarm stuttering or UI frame drops during alerts?
**Answer**:
- The audio playback system in `app/alerts.py` uses asynchronous background threads (`threading.Thread`).
- Sound triggers are decoupled from the main video loop and throttled using a cooldown timer (e.g. 2.0s minimum gap between repeated alarm triggers) so video rendering remains smooth at 30 FPS.

---

### Q3: How is false alarm prevention handled for normal eye blinks?
**Answer**:
- Normal eye blinks last approximately $0.1$ to $0.4$ seconds ($3$–$12$ frames at 30 FPS).
- DriveGuard AI uses temporal frame accumulation: a **Microsleep Alert** is triggered ONLY if the Eye Aspect Ratio (EAR) remains below the threshold for $\ge 20$ consecutive frames ($\sim 0.7\text{s}$).

---

### Q4: How does the Fatigue Score work?
**Answer**:
- It computes a dynamic index from 0 to 100% based on:
  1. Percentage of eye closure time in a rolling 60-second sliding window.
  2. Yawn frequency (yawns per 3-minute window).
  3. Total microsleep penalty accumulation.
  4. Total driving session duration weighting.
