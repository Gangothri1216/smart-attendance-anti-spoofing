"""
Smart Attendance System with Anti-Spoofing
Configuration File
=========================================
Defines all thresholds, file paths, anti-spoofing parameters, and database paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
SPOOF_SNAPSHOTS_DIR = LOGS_DIR / "spoof_attempts"
ATTENDANCE_CSV_DIR = LOGS_DIR / "attendance_csv"

# Ensure runtime directories exist
for folder in [DATASET_DIR, MODELS_DIR, DATABASE_DIR, LOGS_DIR, SPOOF_SNAPSHOTS_DIR, ATTENDANCE_CSV_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_PATH = DATABASE_DIR / "attendance.db"

# Camera Configuration
CAMERA_INDEX = 0             # Default webcam index
FRAME_WIDTH = 640            # Capture width
FRAME_HEIGHT = 480           # Capture height
FPS = 30                     # Target FPS

# Face Detection Parameters
HAAR_FACE_SCALE_FACTOR = 1.15
HAAR_FACE_MIN_NEIGHBORS = 5
HAAR_FACE_MIN_SIZE = (80, 80)
FACE_IMAGE_SIZE = (160, 160)  # Standardized size for feature extraction
SAMPLES_PER_STUDENT = 30     # Number of training face images to capture during registration

# Anti-Spoofing & Liveness Detection Parameters
# 1. Texture & High-Frequency (Laplacian / FFT) Analysis
LAPLACIAN_VAR_THRESHOLD = 60.0       # Blurry/matte prints often have lower variance; screens exhibit abnormal spikes
TEXTURE_FFT_RATIO_LOW = 0.08         # Normal skin frequency band lower bound
TEXTURE_FFT_RATIO_HIGH = 0.88        # Normal skin frequency band upper bound

# 2. Color Space / Subsurface Scattering Check (YCrCb & HSV)
# Real human skin has characteristic Cr & Cb chrominance distribution
CR_MIN = 125
CR_MAX = 178
CB_MIN = 70
CB_MAX = 135
BLUE_CHANNEL_MAX_RATIO = 1.35        # Screen replays often have excessive blue backlight

# 3. Eye-Blink & Motion Tracking
EAR_BLINK_THRESHOLD = 0.21           # Eye Aspect Ratio below this counts as closed
BLINK_CONSEC_FRAMES = 2              # Minimum consecutive frames with eye closed to register a blink
BLINK_WINDOW_SECONDS = 5.0           # Window in which at least 1 blink or micro-motion is required
MIN_MOTION_VARIANCE = 1.2            # Micro-motion / jitter threshold across frames

# 4. Overall Anti-Spoofing Confidence Score Threshold (0 to 100)
# Score >= 65 is deemed REAL (Live human), < 65 is flagged as SPOOF
LIVENESS_PASS_SCORE = 65.0

# Recognition Parameters
RECOGNITION_CONFIDENCE_THRESHOLD = 60.0  # Percentage similarity required for matching (0 to 100)
ATTENDANCE_COOLDOWN_SECONDS = 300        # Cooldown in seconds before the same student can be marked again (5 mins)
DAILY_DUPLICATE_PREVENTION = True       # When True, only marks a student once per day

# Web & Server Settings
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
DEBUG_MODE = False
SECRET_KEY = "smart-attendance-anti-spoofing-major-project-key"
