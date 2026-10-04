"""
Smart Attendance System with Anti-Spoofing
Main Flask Web Application
=========================================
Provides real-time camera streaming, live HUD annotations, biometric liveness verification,
automated attendance logging, cybersecurity threat logs, and analytics.
"""

import cv2
import numpy as np
import time
import os
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_file

import config
from database.db_handler import DatabaseHandler
from core.face_detector import FaceDetector
from core.anti_spoofing import AntiSpoofingEngine
from core.face_recognizer import FaceRecognizer

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Initialize Singletons
db = DatabaseHandler()
detector = FaceDetector()
spoof_engine = AntiSpoofingEngine()
recognizer = FaceRecognizer()

# Global Camera Stream Handler
class CameraStream:
    """Thread-safe camera capture class with automatic fallback to synthetic simulation."""

    def __init__(self, src=config.CAMERA_INDEX):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        self.is_camera_open = self.cap.isOpened()
        
        if self.is_camera_open:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        else:
            print("[Camera Warning] Hardware webcam not found or busy. Initializing simulated demo feed.")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_faces = []
        self.running = True

        # Anti-spoof snapshot cooldown tracker
        self.last_spoof_logged_time = 0.0

    def get_frame(self) -> np.ndarray:
        """Fetches latest frame from hardware camera or generates synthetic feed."""
        if self.is_camera_open:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        # Fallback: Generate sleek dark simulated camera frame
        sim = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        # Background gradient & grid
        for i in range(0, config.FRAME_HEIGHT, 40):
            cv2.line(sim, (0, i), (config.FRAME_WIDTH, i), (20, 25, 35), 1)
        for j in range(0, config.FRAME_WIDTH, 40):
            cv2.line(sim, (j, 0), (j, config.FRAME_HEIGHT), (20, 25, 35), 1)

        cv2.putText(sim, "AegisVision Live Biometric Feed", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
        cv2.putText(sim, "Awaiting Camera / Biometric Target", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
        
        # Animated scanning radar line
        scan_y = int((time.time() * 120) % config.FRAME_HEIGHT)
        cv2.line(sim, (0, scan_y), (config.FRAME_WIDTH, scan_y), (16, 185, 129), 2)

        return sim

    def release(self):
        if self.is_camera_open:
            self.cap.release()

camera = CameraStream()

# ----------------- VIDEO STREAM GENERATOR WITH LIVE HUD -----------------

def generate_frames():
    """MJPEG Stream generator with real-time biometric bounding box & HUD overlays."""
    fps_start = time.time()
    frame_count = 0
    fps_val = 30.0

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            fps_val = 10.0 / (time.time() - fps_start + 1e-6)
            fps_start = time.time()

        # Run Face Detection with 5-point landmarks
        try:
            detections = detector.detect_faces_detailed(frame)
        except Exception:
            detections = []

        annotated = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            landmarks = det["landmarks"]
            x, y, w, h = bbox

            # Extract normalized face ROI
            face_roi = detector.extract_face_roi(frame, bbox)

            # Perform Multi-Modal Anti-Spoofing & Liveness Check
            liveness_res = spoof_engine.check_liveness(frame, bbox, face_roi, landmarks)
            is_live = liveness_res["is_live"]
            liveness_score = liveness_res["liveness_score"]
            attack_type = liveness_res["attack_type"]

            # Visual HUD Color Coding: Green = LIVE, Red = SPOOF
            color = (16, 185, 129) if is_live else (63, 63, 244) # BGR (Green vs Red)

            # Draw sleek HUD corners on face
            corner_len = int(w * 0.2)
            cv2.line(annotated, (x, y), (x + corner_len, y), color, 3)
            cv2.line(annotated, (x, y), (x, y + corner_len), color, 3)
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, 3)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, 3)
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, 3)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, 3)
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, 3)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, 3)

            # Draw facial landmarks
            for pt in landmarks.values():
                cv2.circle(annotated, pt, 3, (56, 189, 248), -1)

            if is_live:
                # Face is verified LIVE -> Perform Recognition
                student_id, conf, is_match = recognizer.recognize(face_roi)

                if is_match:
                    student = db.get_student(student_id)
                    student_name = student["name"] if student else student_id
                    
                    # Mark attendance in database (with duplicate cooldown)
                    marked, msg, rec = db.mark_attendance(student_id, conf, liveness_score)

                    tag_text = f"{student_name} ({conf:.1f}%)"
                    sub_text = f"LIVE | Liveness: {liveness_score:.0f}%"
                else:
                    tag_text = "UNKNOWN (Unregistered)"
                    sub_text = f"LIVE HUMAN | Liveness: {liveness_score:.0f}%"

                # Label Pill
                cv2.rectangle(annotated, (x, max(0, y - 48)), (x + max(180, w), y), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x, max(0, y - 48)), (x + max(180, w), y), color, 1)
                cv2.putText(annotated, tag_text, (x + 8, max(18, y - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(annotated, sub_text, (x + 8, max(36, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            else:
                # SPOOF DETECTED -> Block & Log Threat
                now_ts = time.time()
                if now_ts - camera.last_spoof_logged_time > 10.0:
                    # Save snapshot of spoof attack
                    snap_name = f"spoof_{int(now_ts)}.jpg"
                    snap_path = config.SPOOF_SNAPSHOTS_DIR / snap_name
                    cv2.imwrite(str(snap_path), frame)
                    
                    db.log_spoof_incident(
                        attack_type=attack_type,
                        spoof_score=liveness_score,
                        snapshot_path=snap_name,
                        notes=f"Liveness Check Failed (FFT: {liveness_res['texture_score']}, Color: {liveness_res['color_score']}, Blink: {liveness_res['blink_score']})"
                    )
                    camera.last_spoof_logged_time = now_ts

                tag_text = "SPOOF ATTACK BLOCKED"
                sub_text = f"{attack_type} ({liveness_score:.0f}%)"

                cv2.rectangle(annotated, (x, max(0, y - 48)), (x + max(220, w), y), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x, max(0, y - 48)), (x + max(220, w), y), color, 1)
                cv2.putText(annotated, tag_text, (x + 8, max(18, y - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (63, 63, 244), 2)
                cv2.putText(annotated, sub_text, (x + 8, max(36, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1)

        # Header Watermark & FPS
        cv2.putText(annotated, f"AegisVision Anti-Spoof Engine | FPS: {fps_val:.1f}", (14, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------- WEB ROUTES -----------------

@app.route('/')
def index():
    """Live Attendance Scanner & Real-time Biometric HUD."""
    stats = db.get_dashboard_stats()
    today_logs = db.get_today_attendance()
    return render_template('index.html', stats=stats, today_logs=today_logs)

@app.route('/video_feed')
def video_feed():
    """Video streaming route for MJPEG."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register')
def register():
    """Student Enrollment page with automated face capture."""
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    """Attendance audit logs with date and department filtering."""
    date_filter = request.args.get('date', None)
    dept_filter = request.args.get('department', None)
    search_query = request.args.get('search', None)

    logs = db.get_attendance_logs(date_filter=date_filter, dept_filter=dept_filter, search_query=search_query)
    return render_template('attendance.html', logs=logs, selected_date=date_filter, selected_dept=dept_filter, search_query=search_query)

@app.route('/security')
def security():
    """Cybersecurity & Spoof Incident Logs."""
    incidents = db.get_spoof_incidents(limit=100)
    return render_template('security.html', incidents=incidents)

@app.route('/analytics')
def analytics():
    """Interactive Chart.js visualizations & Attendance Reports."""
    stats = db.get_dashboard_stats()
    
    weekly_dates = [item["date"] for item in stats["weekly_trend"]]
    weekly_counts = [item["count"] for item in stats["weekly_trend"]]
    
    dept_labels = list(stats["dept_counts"].keys()) if stats["dept_counts"] else ["Computer Science", "Cyber Security", "AI & ML", "IT"]
    dept_values = list(stats["dept_counts"].values()) if stats["dept_counts"] else [0, 0, 0, 0]

    return render_template('analytics.html', stats=stats, 
                           weekly_dates=weekly_dates, weekly_counts=weekly_counts,
                           dept_labels=dept_labels, dept_values=dept_values)

@app.route('/students')
def students():
    """Registered Student Directory."""
    student_list = db.get_all_students()
    return render_template('students.html', students=student_list)

# ----------------- REST API ENDPOINTS -----------------

@app.route('/api/live_status')
def api_live_status():
    """Returns today's statistics and recent attendance logs for AJAX polling."""
    stats = db.get_dashboard_stats()
    recent_logs = db.get_today_attendance()
    return jsonify({
        "stats": stats,
        "recent_logs": recent_logs[:15]
    })

@app.route('/api/register_student', methods=['POST'])
def api_register_student():
    """Creates a new student record in the database and creates their dataset directory."""
    data = request.get_json() or {}
    student_id = data.get('student_id', '').strip()
    name = data.get('name', '').strip()
    roll_no = data.get('roll_no', '').strip()
    department = data.get('department', '').strip()
    email = data.get('email', '').strip()
    phone = data.get('phone', '').strip()

    if not student_id or not name or not roll_no or not department:
        return jsonify({"success": False, "message": "Missing required student profile fields."}), 400

    # Ensure student dataset directory exists
    student_dir = config.DATASET_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    success = db.register_student(student_id, name, roll_no, department, email, phone, samples_count=0)
    if success:
        return jsonify({"success": True, "message": f"Student '{name}' registered. Ready for face capture."})
    else:
        return jsonify({"success": False, "message": "Database insertion failed."}), 500

@app.route('/api/capture_sample', methods=['POST'])
def api_capture_sample():
    """Captures a single face crop from the current camera feed for the specified student."""
    data = request.get_json() or {}
    student_id = data.get('student_id', '').strip()

    if not student_id:
        return jsonify({"success": False, "message": "Student ID required."}), 400

    student_dir = config.DATASET_DIR / student_id
    if not student_dir.exists():
        student_dir.mkdir(parents=True, exist_ok=True)

    frame = camera.get_frame()
    if frame is None:
        return jsonify({"success": False, "message": "Camera frame not available."}), 500

    detections = detector.detect_faces_detailed(frame)
    if not detections:
        return jsonify({"success": False, "message": "No face detected in current frame."})

    # Pick the largest detected face
    largest_det = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
    bbox = largest_det["bbox"]
    landmarks = largest_det["landmarks"]

    # Align and crop normalized face
    aligned_face = detector.align_face(frame, landmarks, bbox, target_size=config.FACE_IMAGE_SIZE)

    # Save to student directory
    existing_samples = len(list(student_dir.glob("*.jpg")))
    sample_filename = student_dir / f"face_{existing_samples + 1:03d}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(sample_filename), aligned_face)

    new_count = existing_samples + 1
    # Update sample count in DB
    student = db.get_student(student_id)
    if student:
        db.register_student(student_id, student["name"], student["roll_no"], student["department"], 
                            student["email"], student["phone"], samples_count=new_count)

    return jsonify({
        "success": True,
        "samples_count": new_count,
        "message": f"Sample {new_count} captured successfully."
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    """Triggers dataset training and saves updated recognition model."""
    success, msg = recognizer.train_from_dataset()
    return jsonify({
        "success": success,
        "message": msg
    })

@app.route('/api/export_csv')
def api_export_csv():
    """Downloads master CSV export."""
    csv_file = db.export_all_to_csv()
    return send_file(csv_file, as_attachment=True, download_name=csv_file.name)

@app.route('/api/delete_student/<student_id>', methods=['POST'])
def api_delete_student(student_id):
    """Deletes student from database."""
    success = db.delete_student(student_id)
    if success:
        return jsonify({"success": True, "message": "Student deleted."})
    return jsonify({"success": False, "message": "Delete failed."}), 500

# ----------------- MAIN RUNNER -----------------

if __name__ == '__main__':
    print("=" * 65)
    print(" AegisVision: Smart Attendance System using Anti-Spoofing")
    print(f" Web Server Running at: http://{config.WEB_HOST}:{config.WEB_PORT}")
    print("=" * 65)
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=config.DEBUG_MODE, threaded=True)
