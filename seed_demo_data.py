"""
Demo Data Seeder
================
Generates realistic sample students, synthetic face samples, trained model,
attendance history over the past 7 days, and spoof attack defense logs.
"""

import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import random
import config
from database.db_handler import DatabaseHandler
from core.face_recognizer import FaceRecognizer

def create_synthetic_face(seed_val: int) -> np.ndarray:
    """Generates an organic synthetic facial avatar for feature training."""
    np.random.seed(seed_val)
    img = np.zeros((config.FACE_IMAGE_SIZE[1], config.FACE_IMAGE_SIZE[0], 3), dtype=np.uint8)
    
    # Skin tone background
    base_color = [random.randint(140, 210), random.randint(160, 225), random.randint(200, 250)] # BGR
    img[:] = base_color

    # Add facial gradients & contours
    cv2.circle(img, (80, 80), 65, (base_color[0]-25, base_color[1]-20, base_color[2]-15), -1)
    
    # Eyes
    cv2.ellipse(img, (55, 65), (14, 8), 0, 0, 360, (240, 240, 245), -1)
    cv2.ellipse(img, (105, 65), (14, 8), 0, 0, 360, (240, 240, 245), -1)
    cv2.circle(img, (55, 65), 5, (40, 30, 20), -1)
    cv2.circle(img, (105, 65), 5, (40, 30, 20), -1)

    # Nose
    cv2.line(img, (80, 68), (76, 95), (base_color[0]-40, base_color[1]-35, base_color[2]-30), 2)
    cv2.line(img, (76, 95), (84, 95), (base_color[0]-40, base_color[1]-35, base_color[2]-30), 2)

    # Mouth
    cv2.ellipse(img, (80, 118), (18, 7), 0, 0, 180, (60, 60, 180), -1)

    # Add natural texture noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

def seed_all():
    print("[Seeder] Initializing database and demo dataset...")
    db = DatabaseHandler()

    sample_students = [
        {"student_id": "CS-2026-001", "name": "Aarav Sharma", "roll_no": "22CS01", "department": "Computer Science", "email": "aarav.s@univ.edu", "phone": "+91 98765 43210"},
        {"student_id": "CS-2026-002", "name": "Diya Patel", "roll_no": "22CS02", "department": "Cyber Security", "email": "diya.p@univ.edu", "phone": "+91 98765 43211"},
        {"student_id": "CS-2026-003", "name": "Rohan Verma", "roll_no": "22CS03", "department": "Artificial Intelligence", "email": "rohan.v@univ.edu", "phone": "+91 98765 43212"},
        {"student_id": "CS-2026-004", "name": "Ananya Iyer", "roll_no": "22CS04", "department": "Information Tech", "email": "ananya.i@univ.edu", "phone": "+91 98765 43213"},
        {"student_id": "CS-2026-005", "name": "Kabir Mehta", "roll_no": "22CS05", "department": "Electronics", "email": "kabir.m@univ.edu", "phone": "+91 98765 43214"},
        {"student_id": "CS-2026-006", "name": "Sneha Reddy", "roll_no": "22CS06", "department": "Cyber Security", "email": "sneha.r@univ.edu", "phone": "+91 98765 43215"},
        {"student_id": "CS-2026-007", "name": "Vikram Malhotra", "roll_no": "22CS07", "department": "Computer Science", "email": "vikram.m@univ.edu", "phone": "+91 98765 43216"}
    ]

    for idx, s in enumerate(sample_students):
        # Register in DB
        db.register_student(s["student_id"], s["name"], s["roll_no"], s["department"], s["email"], s["phone"], samples_count=15)
        
        # Create dataset images
        s_dir = config.DATASET_DIR / s["student_id"]
        s_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_i in range(15):
            face_img = create_synthetic_face(idx * 50 + sample_i)
            # Slight random lighting/contrast variations
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            varied = cv2.convertScaleAbs(face_img, alpha=alpha, beta=beta)
            cv2.imwrite(str(s_dir / f"face_{sample_i+1:03d}.jpg"), varied)

    print(f"[Seeder] Created {len(sample_students)} student profiles and training face datasets.")

    # Train model
    recognizer = FaceRecognizer()
    success, msg = recognizer.train_from_dataset()
    print(f"[Seeder] Model Training: {msg}")

    # Seed past 7 days attendance history
    today = datetime.now()
    for day_offset in range(6, -1, -1):
        target_date = today - timedelta(days=day_offset)
        date_str = target_date.strftime("%Y-%m-%d")

        # Pick random subset of students present
        num_present = random.randint(4, len(sample_students))
        present_students = random.sample(sample_students, num_present)

        for s in present_students:
            rand_hour = random.choice([8, 9])
            rand_min = random.randint(15, 55)
            rand_sec = random.randint(10, 59)
            time_str = f"{rand_hour:02d}:{rand_min:02d}:{rand_sec:02d}"
            status = "Late" if (rand_hour == 9 and rand_min > 30) else "Present"
            conf = random.uniform(78.5, 96.8)
            spoof_score = random.uniform(84.0, 98.5)

            with db._get_connection() as conn:
                cursor = conn.cursor()
                # Check if exists
                cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date_str = ?", (s["student_id"], date_str))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO attendance (student_id, name, department, date_str, time_str, status, confidence, spoof_score, liveness_passed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (s["student_id"], s["name"], s["department"], date_str, time_str, status, round(conf, 2), round(spoof_score, 2)))
                    conn.commit()

    print("[Seeder] Created historical attendance audit logs.")

    # Seed Spoof Attack Defense Logs
    spoof_attack_types = [
        "Digital Screen Replay Attack (Mobile/Tablet Display)",
        "Printed Photo / Paper Cutout Attack",
        "Static 2D Portrait Attack (No Natural Blinking)",
        "Digital Screen Replay Attack (Mobile/Tablet Display)"
    ]

    for offset in range(3):
        attack_date = today - timedelta(days=offset)
        for _ in range(random.randint(1, 2)):
            a_type = random.choice(spoof_attack_types)
            score = round(random.uniform(22.0, 48.5), 1)
            time_str = f"{random.randint(9, 16):02d}:{random.randint(10, 50):02d}:{random.randint(10, 59):02d}"
            
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO spoof_incidents (date_str, time_str, attack_type, spoof_score, notes)
                    VALUES (?, ?, ?, ?, ?)
                """, (attack_date.strftime("%Y-%m-%d"), time_str, a_type, score, "Biometric anti-spoofing engine intercepted unauthorized presentation attempt."))
                conn.commit()

    print("[Seeder] Created security spoof incident logs.")
    print("=" * 60)
    print(" Demo Data Successfully Seeded! System is ready to demo.")
    print("=" * 60)

if __name__ == '__main__':
    seed_all()
