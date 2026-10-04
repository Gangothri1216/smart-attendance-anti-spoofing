"""
Face Recognizer & Feature Extraction Module
===========================================
Extracts facial texture representations (LBP Histograms + Multi-Cell Embeddings)
and performs high-accuracy identity classification and confidence calculation.
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.neighbors import KNeighborsClassifier

import config

class FaceRecognizer:
    """Manages facial training datasets, feature extraction, and real-time identity recognition."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (config.MODELS_DIR / "face_recognizer.pkl")
        self.model = None
        self.label_map: Dict[int, str] = {}       # Maps integer label -> student_id
        self.student_names: Dict[str, str] = {}   # Maps student_id -> student_name
        self.is_trained = False

        # Load existing model if available
        self.load_model()

    # ---------------- FEATURE EXTRACTION (LBP & CELL HISTOGRAMS) ----------------

    def compute_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Computes 8-neighborhood Local Binary Pattern (LBP) representation using bitwise shifts.
        LBP is invariant to monotonic grayscale illumination changes.
        """
        h, w = gray_image.shape
        center = gray_image[1:-1, 1:-1]

        # Standard 8-neighborhood LBP computation with fast bitwise operations
        lbp_image = (
            ((gray_image[0:-2, 0:-2] >= center).astype(np.uint8) << 7) |
            ((gray_image[0:-2, 1:-1] >= center).astype(np.uint8) << 6) |
            ((gray_image[0:-2, 2:]   >= center).astype(np.uint8) << 5) |
            ((gray_image[1:-1, 2:]   >= center).astype(np.uint8) << 4) |
            ((gray_image[2:, 2:]     >= center).astype(np.uint8) << 3) |
            ((gray_image[2:, 1:-1]   >= center).astype(np.uint8) << 2) |
            ((gray_image[2:, 0:-2]   >= center).astype(np.uint8) << 1) |
            ((gray_image[1:-1, 0:-2] >= center).astype(np.uint8) << 0)
        )
        return lbp_image

    def extract_features(self, face_bgr_or_gray: np.ndarray, grid_x: int = 8, grid_y: int = 8) -> np.ndarray:
        """
        Extracts multi-cell Spatial LBP histogram embeddings.
        Input is resized to standard dimensions (160x160), converted to LBP, and split into 8x8 cells.
        Each cell generates a normalized 64-bin histogram -> Concatenated into high-dimensional descriptor.
        """
        if len(face_bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr_or_gray

        # Resize to fixed standard
        resized = cv2.resize(gray, config.FACE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        # Apply histogram equalization for lighting stability
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_gray = clahe.apply(resized)

        # Compute LBP texture map
        lbp = self.compute_lbp(norm_gray)
        h, w = lbp.shape
        cell_h, cell_w = h // grid_y, w // grid_x

        histograms = []
        for i in range(grid_y):
            for j in range(grid_x):
                cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=64, range=(0, 256))
                # Normalize cell histogram (L2 norm)
                hist = hist.astype(np.float32)
                norm_val = np.linalg.norm(hist) + 1e-6
                hist /= norm_val
                histograms.extend(hist)

        feature_vector = np.array(histograms, dtype=np.float32)
        # Global unit-norm normalization
        global_norm = np.linalg.norm(feature_vector) + 1e-6
        feature_vector /= global_norm
        return feature_vector

    # ---------------- TRAINING & PERSISTENCE ----------------

    def train_from_dataset(self, dataset_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Scans dataset directory, extracts facial embeddings for all registered students,
        and trains a k-Nearest Neighbors / Cosine Classifier.
        """
        ds_dir = dataset_dir or config.DATASET_DIR
        if not ds_dir.exists():
            return False, "Dataset directory does not exist."

        student_folders = [f for f in ds_dir.iterdir() if f.is_dir()]
        if not student_folders:
            return False, "No student image folders found in dataset."

        X_train = []
        y_train = []
        self.label_map = {}
        self.student_names = {}

        label_counter = 0

        for folder in student_folders:
            student_id = folder.name
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            
            if len(image_files) < 3:
                # Minimum 3 images required to ensure robust representation
                continue

            self.label_map[label_counter] = student_id
            
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                feats = self.extract_features(img)
                X_train.append(feats)
                y_train.append(label_counter)

            label_counter += 1

        if not X_train or label_counter == 0:
            return False, "Insufficient face samples found for training. Please register students with at least 5 face captures."

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train Cosine Distance KNN Classifier
        # Metric='cosine' ensures scale and illumination-invariant angle matching
        knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train)), metric='cosine', weights='distance')
        knn.fit(X_train, y_train)

        self.model = knn
        self.is_trained = True

        # Save to disk
        self.save_model()
        return True, f"Successfully trained recognition model with {len(X_train)} samples across {label_counter} students."

    def save_model(self):
        """Persists trained classifier and metadata to disk."""
        data = {
            "model": self.model,
            "label_map": self.label_map,
            "student_names": self.student_names,
            "is_trained": self.is_trained
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self) -> bool:
        """Loads classifier from disk if available."""
        if not self.model_path.exists():
            self.is_trained = False
            return False

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data.get("model")
                self.label_map = data.get("label_map", {})
                self.student_names = data.get("student_names", {})
                self.is_trained = data.get("is_trained", False)
                return True
        except Exception as e:
            print(f"[Model Load Warning] {e}")
            self.is_trained = False
            return False

    # ---------------- REAL-TIME INFERENCE & MATCHING ----------------

    def recognize(self, face_bgr_or_gray: np.ndarray) -> Tuple[str, float, bool]:
        """
        Recognizes identity of given face image.
        Returns: (student_id: str, confidence_score: float [0-100], is_recognized: bool)
        """
        if not self.is_trained or self.model is None or len(self.label_map) == 0:
            return "UNKNOWN (Untrained)", 0.0, False

        feats = self.extract_features(face_bgr_or_gray).reshape(1, -1)

        # Find nearest neighbor distances
        distances, indices = self.model.kneighbors(feats, n_neighbors=1)
        cosine_dist = distances[0][0]  # Distance between 0.0 (identical) and 2.0 (opposite)
        predicted_label = self.model.predict(feats)[0]

        # Convert cosine distance to percentage similarity
        # Cosine distance of 0.15 is high similarity (~92%), > 0.45 is unknown
        similarity = max(0.0, (1.0 - (cosine_dist / 0.50))) * 100.0
        similarity = float(np.clip(similarity, 0.0, 100.0))

        student_id = self.label_map.get(predicted_label, "UNKNOWN")
        is_match = (similarity >= config.RECOGNITION_CONFIDENCE_THRESHOLD)

        if not is_match:
            return "UNKNOWN", similarity, False

        return student_id, similarity, True
