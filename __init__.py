"""
Core vision and recognition package initialization.
"""
from core.face_detector import FaceDetector
from core.anti_spoofing import AntiSpoofingEngine
from core.face_recognizer import FaceRecognizer

__all__ = ["FaceDetector", "AntiSpoofingEngine", "FaceRecognizer"]
