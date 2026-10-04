@echo off
title AegisVision - Smart Attendance System with Anti-Spoofing
color 0b
echo ======================================================================
echo   AegisVision: Smart Attendance System using Anti-Spoofing
echo ======================================================================
echo.
echo Starting Web Server and opening browser...
echo.
start "" "http://127.0.0.1:5000"
python app.py
pause
