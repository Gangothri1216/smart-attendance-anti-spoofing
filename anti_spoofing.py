"""
Anti-Spoofing & Multi-Modal Liveness Detection Engine
=====================================================
Combines 4 distinct computer vision and physical biometric security layers:
1. Texture & Moiré Frequency Spectrum (2D-FFT Radial Distribution + Laplacian Edge Analysis)
   - Rejects digital LCD/OLED screens (which exhibit high-frequency pixel raster grids and moiré artifacts).
   - Rejects printed photos and matte paper (which show flat low-frequency grain or unnatural blur).
2. Chrominance & Subsurface Scattering (YCrCb Color Space + Blue Channel Glare Ratio)
   - Real skin absorbs and scatters light across capillary blood vessels (Cr/Cb color bounds).
   - Electronic displays emit strong polarized blue LED backlight (Blue intensity ratio > 1.35).
3. Eye Dynamics & Blink State Machine (Landmark Eye Aspect Ratio / Gradient Aperture)
   - Tracks natural blink transitions (Open -> Closed -> Open).
   - Prevents static 2D portrait photos and cutout paper attacks.
4. Micro-Motion & Centroid Variance (Breathing & Head Jitter Tracking)
   - Measures natural involuntary physiological micro-tremors.
   - Flags completely frozen images mounted on static stands.
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import Dict, Tuple, Any, Optional

import config

class AntiSpoofingEngine:
    """Evaluates biometric liveness to prevent proxy and spoofing attacks."""

    def __init__(self):
        # Temporal buffers for motion and blink state tracking
        self.eye_state_history = deque(maxlen=40)       # (timestamp, avg_openness)
        self.centroid_history = deque(maxlen=30)        # (timestamp, cx, cy)
        self.last_blink_time = time.time()
        self.consecutive_closed_frames = 0
        self.total_blinks_detected = 0

    # ---------------- 1. TEXTURE & FREQUENCY SPECTRUM (FFT & LAPLACIAN) ----------------

    def analyze_texture_fft(self, face_roi_gray: np.ndarray) -> Tuple[float, float]:
        """
        Calculates 2D Discrete Fourier Transform (FFT) power distribution.
        Real faces exhibit balanced frequency dissipation.
        Phone screens & printed photos exhibit artificial moiré spikes or flat high-frequency suppression.
        Returns: (fft_score: float [0-100], high_freq_ratio: float)
        """
        if face_roi_gray.size == 0 or face_roi_gray.shape[0] < 10:
            return 0.0, 0.0

        f = np.fft.fft2(face_roi_gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-6)

        h, w = face_roi_gray.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        radius = min(cx, cy) // 2

        low_freq_energy = np.mean(magnitude[dist_from_center <= radius])
        high_freq_energy = np.mean(magnitude[dist_from_center > radius])

        ratio = float(high_freq_energy / (low_freq_energy + 1e-6))

        # Human skin ratio usually falls within [0.10, 0.85]
        if config.TEXTURE_FFT_RATIO_LOW <= ratio <= config.TEXTURE_FFT_RATIO_HIGH:
            fft_score = 95.0 - abs(ratio - 0.45) * 65.0
        else:
            fft_score = max(10.0, 60.0 - abs(ratio - 0.45) * 90.0)

        fft_score = float(np.clip(fft_score, 0.0, 100.0))
        return fft_score, ratio

    def analyze_laplacian_focus(self, face_roi_gray: np.ndarray) -> Tuple[float, float]:
        """
        Measures Laplacian edge variance to detect paper blur or digital pixel edges.
        Returns: (laplacian_score: float [0-100], variance: float)
        """
        if face_roi_gray.size == 0 or face_roi_gray.shape[0] < 10:
            return 0.0, 0.0

        laplacian = cv2.Laplacian(face_roi_gray, cv2.CV_64F)
        var = float(laplacian.var())

        if var < config.LAPLACIAN_VAR_THRESHOLD:
            # Low focus / blurred printed photo
            score = max(15.0, (var / config.LAPLACIAN_VAR_THRESHOLD) * 60.0)
        elif var > 850.0:
            # Harsh digital screen subpixel raster noise
            score = max(20.0, 100.0 - (var - 850.0) * 0.08)
        else:
            score = 92.0

        score = float(np.clip(score, 0.0, 100.0))
        return score, var

    # ---------------- 2. COLOR SPACE & BACKLIGHT REFLECTANCE ----------------

    def analyze_color_chrominance(self, face_roi_bgr: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Analyzes YCrCb skin chrominance cluster and detects digital screen blue-light saturation.
        Returns: (color_score: float [0-100], details: dict)
        """
        if face_roi_bgr.size == 0 or face_roi_bgr.shape[0] < 10:
            return 0.0, {}

        # 1. YCrCb Skin Model
        ycrcb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        mean_cr = float(np.mean(cr))
        mean_cb = float(np.mean(cb))

        cr_valid = (config.CR_MIN <= mean_cr <= config.CR_MAX)
        cb_valid = (config.CB_MIN <= mean_cb <= config.CB_MAX)

        skin_match_score = 100.0 if (cr_valid and cb_valid) else 35.0

        # 2. Blue Light Screen Glare Ratio
        b, g, r = cv2.split(face_roi_bgr.astype(np.float32))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        
        total_intensity = avg_b + avg_g + avg_r + 1e-6
        blue_ratio = (avg_b * 3.0) / total_intensity

        if blue_ratio > config.BLUE_CHANNEL_MAX_RATIO:
            # Phone / Monitor LED screen detected
            backlight_score = max(10.0, 100.0 - (blue_ratio - config.BLUE_CHANNEL_MAX_RATIO) * 160.0)
        else:
            backlight_score = 95.0

        color_score = (skin_match_score * 0.5) + (backlight_score * 0.5)
        color_score = float(np.clip(color_score, 0.0, 100.0))

        details = {
            "mean_cr": round(mean_cr, 1),
            "mean_cb": round(mean_cb, 1),
            "blue_ratio": round(float(blue_ratio), 2),
            "color_score": round(color_score, 1)
        }
        return color_score, details

    # ---------------- 3. EYE DYNAMICS & BLINK TRACKING ----------------

    def compute_eye_openness(self, frame_bgr: np.ndarray, landmarks: Optional[Dict[str, Tuple[int, int]]]) -> float:
        """
        Extracts eye regions from landmarks and computes vertical aperture gradient.
        Returns: openness_ratio: float (0.0 = closed, 1.0 = open)
        """
        if not landmarks or "right_eye" not in landmarks or "left_eye" not in landmarks:
            return 0.5

        h_frame, w_frame = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        openness_list = []
        for eye_key in ["right_eye", "left_eye"]:
            ex, ey = landmarks[eye_key]
            # Crop 24x24 eye region around landmark
            ew, eh = 18, 14
            x1, y1 = max(0, ex - ew), max(0, ey - eh)
            x2, y2 = min(w_frame, ex + ew), min(h_frame, ey + eh)

            eye_crop = gray[y1:y2, x1:x2]
            if eye_crop.size < 50:
                continue

            # Compute vertical edge intensity gradient vs horizontal
            grad_y = cv2.Sobel(eye_crop, cv2.CV_64F, 0, 1, ksize=3)
            vert_intensity = np.mean(np.abs(grad_y))
            # Open eyes have high contrast between dark iris/pupil and white sclera
            openness = min(1.0, vert_intensity / 40.0)
            openness_list.append(openness)

        if not openness_list:
            return 0.5
        return float(np.mean(openness_list))

    def evaluate_eye_dynamics(self, frame_bgr: np.ndarray, landmarks: Optional[Dict[str, Tuple[int, int]]]) -> Tuple[float, bool]:
        """
        Tracks eye state transitions to verify biological blinking.
        Returns: (blink_score: float [0-100], is_blinking: bool)
        """
        now = time.time()
        openness = self.compute_eye_openness(frame_bgr, landmarks)

        # Openness < 0.22 counts as closed eye during blink
        is_closed = (openness < config.EAR_BLINK_THRESHOLD)

        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            if 1 <= self.consecutive_closed_frames <= 6:
                # Valid physiological blink detected
                self.total_blinks_detected += 1
                self.last_blink_time = now
            self.consecutive_closed_frames = 0

        self.eye_state_history.append((now, openness))
        time_since_blink = now - self.last_blink_time

        # Scoring
        if time_since_blink <= config.BLINK_WINDOW_SECONDS or self.total_blinks_detected > 0:
            blink_score = 96.0
        elif time_since_blink <= config.BLINK_WINDOW_SECONDS * 2.0:
            blink_score = max(50.0, 96.0 - (time_since_blink - config.BLINK_WINDOW_SECONDS) * 8.0)
        else:
            # Static photo presentation with 0 blinks
            blink_score = 30.0

        return float(blink_score), (self.consecutive_closed_frames > 0)

    # ---------------- 4. MICRO-MOTION & CENTROID DYNAMICS ----------------

    def evaluate_micro_motion(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Tracks face centroid micro-movements to reject perfectly static cardboard/prints.
        Returns: motion_score: float [0-100]
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        now = time.time()

        self.centroid_history.append((now, cx, cy))

        if len(self.centroid_history) < 6:
            return 85.0

        centroids = np.array([[c[1], c[2]] for c in self.centroid_history])
        std_motion = float(np.std(centroids))

        if std_motion < 0.12:
            # Perfectly static printed photo mounted on stand
            motion_score = 30.0
        elif std_motion > 45.0:
            # Rapid screen jitter / shaking phone
            motion_score = 50.0
        else:
            # Natural micro-motion of living human
            motion_score = 92.0

        return motion_score

    # ---------------- 5. FUSED MULTI-LAYER ASSESSMENT ----------------

    def check_liveness(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], 
                       face_roi_bgr: np.ndarray, landmarks: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Runs full 4-layer liveness pipeline.
        Returns complete diagnostic dictionary.
        """
        face_roi_gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)

        # Layer 1: Frequency & Texture
        fft_score, fft_ratio = self.analyze_texture_fft(face_roi_gray)
        laplacian_score, laplacian_var = self.analyze_laplacian_focus(face_roi_gray)
        texture_score = (fft_score * 0.6) + (laplacian_score * 0.4)

        # Layer 2: Color Space & Subsurface Reflectance
        color_score, color_details = self.analyze_color_chrominance(face_roi_bgr)

        # Layer 3: Eye Dynamics & Blink State Machine
        blink_score, is_blinking = self.evaluate_eye_dynamics(frame_bgr, landmarks)

        # Layer 4: Micro-Motion Variance
        motion_score = self.evaluate_micro_motion(bbox)

        # Comprehensive Weighted Fusion
        overall_score = (
            (texture_score * 0.30) +
            (color_score * 0.25) +
            (blink_score * 0.30) +
            (motion_score * 0.15)
        )
        overall_score = float(np.clip(overall_score, 0.0, 100.0))
        is_live = (overall_score >= config.LIVENESS_PASS_SCORE)

        # Classify Attack Type if detected
        attack_type = "REAL / LIVE HUMAN"
        if not is_live:
            if color_details.get("blue_ratio", 1.0) > config.BLUE_CHANNEL_MAX_RATIO or fft_score < 45.0:
                attack_type = "Digital Screen Replay Attack (Phone/Tablet Display)"
            elif laplacian_score < 40.0 or motion_score < 40.0:
                attack_type = "Printed Photo / Paper Cutout Attack"
            elif blink_score < 45.0:
                attack_type = "Static 2D Portrait Attack (No Natural Blinking)"
            else:
                attack_type = "Synthetic / Spoofed Face Presentation"

        return {
            "is_live": is_live,
            "liveness_score": round(overall_score, 1),
            "attack_type": attack_type,
            "texture_score": round(texture_score, 1),
            "color_score": round(color_score, 1),
            "blink_score": round(blink_score, 1),
            "motion_score": round(motion_score, 1),
            "fft_ratio": round(fft_ratio, 3),
            "laplacian_var": round(laplacian_var, 1),
            "is_blinking": is_blinking,
            "blinks_count": self.total_blinks_detected,
            "color_details": color_details
        }
