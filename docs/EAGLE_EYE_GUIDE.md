# Eagle Eye AI — Intelligent Security Surveillance System

Eagle Eye AI is an enterprise-grade upgrade to the Sentinel Vision computer vision architecture, extending it into a proactive, multi-model safety and threat detection surveillance solution.

---

## Key Capabilities

1. **Dual-Model Inference Pipeline**:
   - **Object & Person Detection**: Runs Ultralytics YOLO11 Nano (`yolo11n.pt`) with ByteTrack / BoTSORT tracking for people, vehicles, backpacks, suitcases, etc.
   - **Dedicated Weapon Detection**: Runs custom YOLO weapon model (`models/rik_best_best.pt`) targeting **pistols** and **knives**.
   - Model class labels inspected dynamically (`{0: 'pistol', 1: 'knife'}`).

2. **False-Positive Mitigation via Temporal Confirmation**:
   - Weapons are not alerted on single-frame noise. Detections require consecutive spatial overlap (IoU $\ge 0.3$) across multiple inference evaluation windows (`confirmation_frames: 2`).
   - Frame-skipping (`inference_interval: 3`) reduces CPU compute while maintaining continuous weapon safety.
   - Spatial association matches weapons to nearby/enclosing person tracks.

3. **Audible Threat Alarm (Security Beep & Emergency Siren)**:
   - Built on native non-blocking Windows audio (`winsound.Beep`).
   - **Security Beep**: Clear warning tone (1050 Hz) for perimeter line breaches, loitering, and restricted zone intrusions.
   - **Emergency Siren**: High-priority alternating dual-frequency tone (1550 Hz / 850 Hz) for confirmed weapon threats.
   - Thread-safe background priority worker with weapon preemption (siren immediately clears/overrides ordinary beeps).
   - Cooldown and rate limiting prevents runaway audio loops.

4. **Incident Evidence Recording**:
   - Automatic pre-event buffer (3s) and post-event window (3s) recording into high-quality MP4 clips and timestamped snapshots.
   - JSON metadata saved with ISO-8601 UTC timestamp, camera ID, zone, weapon category, and confidence score.

5. **Live Camera Visualization (OpenCV)**:
   - High-contrast **EAGLE EYE AI** banner with real-time FPS counter.
   - Status indicators:
     - `MONITORING — NO ACTIVE ALERT` (Green)
     - `SECURITY WARNING — RESTRICTED AREA INTRUSION` (Amber)
     - `CRITICAL — POSSIBLE WEAPON DETECTED` (Vibrant Red)
   - Red weapon bounding boxes with solid background labels for high projector visibility.

6. **Centralized Surveillance Dashboard**:
   - Streamlit interface with KPI metrics: Total Incidents, Weapon Detections, High/Critical Alerts.
   - Filters by Camera, Zone, Event Type, and Severity (`critical`, `high`, `medium`, `low`).
   - Built-in Incident Evidence Inspector with snapshot preview and video playback.

---

## Quick Start Commands

### 1. Run Live Camera Surveillance
```cmd
python -m src.main --config configs/default.yaml --profile eagle_eye_cpu
```
*(Press `q` inside the video window to stop)*

### 2. Audio Verification Test
```cmd
python -m src.utils.security_audio --test
```

### 3. Launch Surveillance Dashboard
```cmd
streamlit run src/api/dashboard.py
```

### 4. Launch FastAPI Alerts Backend
```cmd
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Run Automated Test Suite
```cmd
pytest tests
```
*(62 tests passing)*
