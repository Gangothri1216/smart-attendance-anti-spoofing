"""
Face Detector Module (Deep Learning YuNet + 5 Facial Landmarks)
==============================================================
Provides high-speed, multi-angle face detection and precise 5-point facial landmarking
(Right Eye, Left Eye, Nose Tip, Right Mouth Corner, Left Mouth Corner) using OpenCV YuNet DNN.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import config

class FaceDetector:
    """Deep learning face detector utilizing OpenCV YuNet ONNX with 5-point landmark localization."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (config.MODELS_DIR / "face_detection_yunet.onnx")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"YuNet model not found at {self.model_path}")

        # Initialize YuNet Detector
        self.input_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=self.input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=10
        )

        # CLAHE for lighting equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def set_input_size(self, width: int, height: int):
        """Updates internal detector resolution to match input frame dimensions."""
        if (width, height) != self.input_size:
            self.input_size = (width, height)
            self.detector.setInputSize(self.input_size)

    def detect_faces_detailed(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in frame and returns bounding boxes, 5-point landmarks, and confidence.
        Returns list of dicts:
        {
            'bbox': (x, y, w, h),
            'landmarks': {
                'right_eye': (x, y),
                'left_eye': (x, y),
                'nose': (x, y),
                'right_mouth': (x, y),
                'left_mouth': (x, y)
            },
            'confidence': float
        }
        """
        h_frame, w_frame = frame.shape[:2]
        self.set_input_size(w_frame, h_frame)

        # Detect
        _, detections = self.detector.detect(frame)

        results = []
        if detections is None or len(detections) == 0:
            return results

        for det in detections:
            # Format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            
            # Clip bounding box within frame
            x = max(0, x)
            y = max(0, y)
            w = min(w_frame - x, w)
            h = min(h_frame - y, h)

            if w < 20 or h < 20:
                continue

            landmarks = {
                "right_eye": (int(det[4]), int(det[5])),
                "left_eye": (int(det[6]), int(det[7])),
                "nose": (int(det[8]), int(det[9])),
                "right_mouth": (int(det[10]), int(det[11])),
                "left_mouth": (int(det[12]), int(det[13]))
            }
            score = float(det[14])

            results.append({
                "bbox": (x, y, w, h),
                "landmarks": landmarks,
                "confidence": score
            })

        return results

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Convenience method returning list of bounding boxes (x, y, w, h)."""
        detailed = self.detect_faces_detailed(frame)
        return [d["bbox"] for d in detailed]

    def extract_face_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                         target_size: Tuple[int, int] = config.FACE_IMAGE_SIZE) -> np.ndarray:
        """Crops and standardizes face crop to target_size."""
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0 or face_roi.shape[0] < 5 or face_roi.shape[1] < 5:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        return resized

    def align_face(self, frame: np.ndarray, landmarks: Dict[str, Tuple[int, int]], 
                   bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = config.FACE_IMAGE_SIZE) -> np.ndarray:
        """
        Aligns the face horizontally based on the angle between the two eyes.
        Ensures illumination and pose consistency for feature matching.
        """
        r_eye = landmarks["right_eye"]
        l_eye = landmarks["left_eye"]

        # Calculate angle between eyes
        dy = l_eye[1] - r_eye[1]
        dx = l_eye[0] - r_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Face center
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)

        # Rotate matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h_frame, w_frame = frame.shape[:2]
        rotated = cv2.warpAffine(frame, M, (w_frame, h_frame), flags=cv2.INTER_CUBIC)

        # Crop face from rotated image
        return self.extract_face_roi(rotated, bbox, target_size)
