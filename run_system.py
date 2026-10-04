"""
AegisVision - Smart Attendance System with Anti-Spoofing
Unified Master Launcher
======================================================
Interactive console launcher to run Web Dashboard, Desktop GUI, Seed Demo Data, or Diagnostics.
"""

import os
import sys
import subprocess
import time

def print_banner():
    print("""
======================================================================
  🛡️  AEGISVISION: SMART ATTENDANCE SYSTEM USING ANTI-SPOOFING  🛡️
         Cybersecurity & Biometric Liveness Verification AI
======================================================================
    """)

def run_web_dashboard():
    print("\n[+] Launching Web Server & Interactive Dashboard...")
    print("[+] Opening browser at: http://127.0.0.1:5000")
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass
    import app
    app.app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

def run_desktop_gui():
    print("\n[+] Launching Desktop Tkinter GUI...")
    import main_gui
    main_gui.launch_gui()

def run_seeder():
    print("\n[+] Seeding sample students, attendance history, and spoof logs...")
    import seed_demo_data
    seed_demo_data.seed_all()

def run_diagnostics():
    print("\n[+] Running Comprehensive System Self-Test...")
    try:
        import cv2, numpy, flask, PIL, matplotlib, pandas, sklearn
        from database import DatabaseHandler
        from core import FaceDetector, AntiSpoofingEngine, FaceRecognizer

        db = DatabaseHandler()
        students_count = db.count_students()
        logs_count = len(db.get_attendance_logs())
        spoofs_count = len(db.get_spoof_incidents())
        
        fd = FaceDetector()
        as_eng = AntiSpoofingEngine()
        fr = FaceRecognizer()

        print(f" [✓] Database initialized ({students_count} students, {logs_count} attendance records, {spoofs_count} spoof alerts)")
        print(" [✓] Deep Learning Face Detector (YuNet 5-point landmark ONNX) loaded")
        print(" [✓] Multi-Modal Anti-Spoofing Engine (2D-FFT + Laplacian + YCrCb + Blink) loaded")
        print(f" [✓] Face Recognizer Classifier (Spatial LBP + Cosine KNN) loaded (Is Trained: {fr.is_trained})")
        print("\n>>> ALL SYSTEM DIAGNOSTICS PASSED PERFECTLY! <<<")
    except Exception as e:
        print(f"\n[!] Diagnostic Error: {e}")

def main():
    while True:
        print_banner()
        print("Select an option to run:")
        print(" [1] Launch Web Dashboard (Recommended - Full UI, Real-time Stream & Analytics)")
        print(" [2] Launch Desktop GUI Application (Tkinter Window)")
        print(" [3] Re-Seed Demo Dataset & Train Model (Fresh sample students & logs)")
        print(" [4] Run System Diagnostics & Integrity Check")
        print(" [5] Exit")
        print("-" * 70)

        choice = input("Enter choice (1-5) [Default=1]: ").strip() or "1"

        if choice == "1":
            run_web_dashboard()
            break
        elif choice == "2":
            run_desktop_gui()
            break
        elif choice == "3":
            run_seeder()
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            run_diagnostics()
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            print("\nExiting AegisVision. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option. Please choose 1 to 5.")
            time.sleep(1)

if __name__ == '__main__':
    main()
