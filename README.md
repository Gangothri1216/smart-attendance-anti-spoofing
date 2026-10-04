# 🛡️ AegisVision: Smart Attendance System using Anti-Spoofing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20DNN-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-red.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, production-grade **Smart Attendance System** that solves proxy attendance and presentation attacks (spoofing) using multi-modal biometric liveness detection and deep learning face recognition.

---

## 🌟 Key Features

- **Deep Learning Face Detection (YuNet DNN)**: Multi-angle face localization and 5-point facial landmarking (eyes, nose, mouth corners) running at 30+ FPS.
- **Multi-Modal Anti-Spoofing & Liveness Engine**:
  - 📡 **2D Fast Fourier Transform (FFT)**: Identifies digital screen pixel rasterization and moiré patterns.
  - 🔍 **Laplacian Edge Variance**: Differentiates blurry paper prints from in-focus human faces.
  - 🎨 **YCrCb Color Space & Backlight Ratio**: Validates organic skin chrominance vs. blue LED screen backlights.
  - 👁️ **Dynamic Eye-Blink State Machine**: Tracks natural physiological blinks to stop static 2D photo attacks.
  - 🧬 **Micro-Motion Variance**: Tracks involuntary physiological head motion to reject fixed cardboard stands.
- **Spatial Local Binary Pattern (LBP) Recognition**: Illuminance-invariant facial texture embedding with Cosine-distance classification.
- **Anti-Proxy Database & Cooldown Management**: Prevents duplicate attendance logging within the same session or day.
- **Full-Stack Interfaces**:
  - 🌐 **Web Dashboard (Flask + Bootstrap 5 + Chart.js)**: Live video stream with biometric HUD telemetry, interactive registration, attendance audit logs, and security threat logs.
  - 🖥️ **Desktop Application (Tkinter GUI)**: Standalone kiosk interface for offline terminal deployments.
- **Cybersecurity & Threat Forensics**: Automatically logs intercepted presentation attacks with timestamps, attack classification, and captured snapshots.

---

## 📁 Project Structure

```
project_3/
├── app.py                      # Flask Web Dashboard & Real-Time MJPEG Stream
├── main_gui.py                 # Standalone Desktop Tkinter GUI Application
├── run_system.py               # Master Unified Launcher with Interactive Menu
├── seed_demo_data.py           # Pre-populates sample students, attendance, & spoof logs
├── config.py                   # Global system parameters, thresholds, and paths
├── requirements.txt            # Python dependencies
├── PROJECT_REPORT.md           # Complete Major Project Academic Documentation
├── INTERVIEW_VIVA_GUIDE.md     # 35+ Interview & Viva Voce Q&A + Pitch Script
│
├── core/                       # Core Computer Vision & AI Modules
│   ├── face_detector.py        # YuNet DNN face detection & 5-point landmark alignment
│   ├── anti_spoofing.py        # Multi-modal liveness defense (FFT, Color, Blink, Motion)
│   └── face_recognizer.py      # Spatial LBP embedding & Cosine classifier
│
├── database/                   # Database & Storage Layer
│   ├── db_handler.py           # SQLite handler for Students, Attendance, & Threats
│   └── attendance.db           # SQLite database
│
├── models/                     # Trained weights & classifiers
│   ├── face_detection_yunet.onnx # OpenCV YuNet DNN face model
│   └── face_recognizer.pkl     # Trained student recognition model
│
├── dataset/                    # Enrolled student face image datasets
│   └── [student_id]/
│
├── logs/                       # Daily CSV attendance exports & spoof snapshots
│   ├── attendance_csv/
│   └── spoof_attempts/
│
├── templates/                  # Web HTML5 Templates
│   ├── base.html               # Base layout with navbar & real-time clock
│   ├── index.html              # Live Scanner with real-time biometric HUD
│   ├── register.html           # Student Biometric Enrollment with live 30-shot capture
│   ├── attendance.html         # Attendance audit table with search, filter, & CSV download
│   ├── security.html           # Anti-Spoofing & Cybersecurity threat logs
│   ├── analytics.html          # Interactive Chart.js graphs and attendance rates
│   └── students.html           # Registered student directory
│
└── static/                     # Static CSS & JS Assets
    ├── css/style.css           # Modern cybersecurity-themed dark stylesheet
    └── js/main.js
```

---

## 🚀 Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the System
Run the unified interactive launcher:
```bash
python run_system.py
```
Select:
- `[1]` to launch the **Web Dashboard** (Opens at `http://127.0.0.1:5000`)
- `[2]` to launch the **Desktop GUI Application**
- `[3]` to re-seed sample demo students and historical attendance logs
- `[4]` to run system self-test diagnostics

---

## 🎯 Live Workflow Demonstration

1. **Live Attendance Scanning**: Open `http://127.0.0.1:5000`. The camera feed will detect faces, evaluate liveness in real time, and mark attendance once verified.
2. **Biometric Student Enrollment**: Navigate to **"Register Student"**, enter student details, and click **"Start Automated 30-Shot Face Capture"**. The system captures multi-angle facial crops and trains the model immediately.
3. **Spoof Attack Interception**: Present a smartphone with a photo or a printed picture to the camera. The system will flag a **Red HUD Warning ("SPOOF ATTACK BLOCKED")**, deny access, and record the incident under **"Security & Spoof Logs"**.
4. **Reports & Master CSV Export**: Visit **"Attendance Logs"** and click **"Export Master CSV"** to generate an attendance report.

---

## 📚 Academic Deliverables Included

- **[Full Major Project Report (PROJECT_REPORT.md)](PROJECT_REPORT.md)**: IEEE-style documentation containing problem definition, mathematical formulations (2D-FFT, Laplacian, LBP, Cosine similarity), architecture diagrams, performance evaluation, and references.
- **[Interview & Viva Voce Preparation Guide (INTERVIEW_VIVA_GUIDE.md)](INTERVIEW_VIVA_GUIDE.md)**: 30-second/90-second pitch scripts, 30+ technical questions & answers covering Computer Vision, Anti-Spoofing, and Cybersecurity, plus defense against tricky examiner questions.
