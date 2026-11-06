# Real-Time Driver Drowsiness Detection System

Real-Time Driver Drowsiness Detection using Facial Landmarks, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), 3D Head Pose Estimation, and a Spatial-Temporal Deep Learning Model (CNN + Bidirectional LSTM).

---

## 📌 Features
- **Real-Time Landmark Tracking**: Extracts 468 facial points using MediaPipe Face Mesh.
- **Biometric Ratio Formulation**:
  - Eye Aspect Ratio (EAR) for blink & micro-sleep detection.
  - Mouth Aspect Ratio (MAR) for yawning detection.
  - 3D Head Pose (Pitch, Yaw, Roll) via `cv2.solvePnP` for forward head nod detection.
  - PERCLOS (% of eye closure time).
- **CNN + LSTM Deep Learning**:
  - TimeDistributed Conv2D spatial feature extractor.
  - Bidirectional LSTM temporal sequence classifier (Alert, Mild Drowsiness, Severe Drowsiness).
- **Emergency Acoustic Alarm**: Auto-triggers buzzer on sustained micro-sleep.
- **Dual Operating Modes**: Live webcam stream & offline recorded video file analysis.
- **Audit Logging**: Persists telemetry records to SQLite and CSV.

---

## 🛠️ Installation & Setup

### 1. Clone the repository and navigate to project root
```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Create and activate a Python virtual environment
```bash
# On Linux / macOS
python3.12 -m venv venv
source venv/bin/activate

# On Windows
py -3.12 -m venv venv
venv\Scripts\activate
```

Python 3.12 is required because the pinned MediaPipe version uses the legacy
`solutions.face_mesh` API and is not available for Python 3.13.

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Launch the Streamlit Web Application
```bash
streamlit run app_streamlit.py
```
The application will open automatically at `http://localhost:8501`.

### 2. Collect Custom Webcam Dataset
To record custom video frames with your own webcam:
```bash
python data_collection.py
```
- Press `a` to record **Alert** frames.
- Press `m` to record **Mild Drowsiness** (yawning/fatigued) frames.
- Press `s` to record **Severe Drowsiness** (micro-sleep/closed eyes) frames.
- Press `q` to stop and save.

### 3. Train the CNN-LSTM Model
```bash
python train.py
```
Trained model checkpoints will be saved in `saved_models/best_drowsiness_cnn_lstm.keras`.

### 4. Evaluate Model Performance
```bash
python evaluate.py
```
Generates `confusion_matrix.png`, `roc_curves.png`, classification metrics report, and FPS benchmarks.
