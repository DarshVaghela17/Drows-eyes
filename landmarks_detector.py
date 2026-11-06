"""
Real-Time Facial Landmark Detector & Metric Calculation Engine
Uses MediaPipe Face Landmarker (Tasks API) with 478 landmarks to compute:
- Eye Aspect Ratio (EAR) for left and right eyes
- Mouth Aspect Ratio (MAR) for yawning detection
- 3D Head Pose (Pitch, Yaw, Roll) via cv2.solvePnP
- Dynamic Blink Counter and PERCLOS
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


class FacialLandmarkDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Locate model file relative to this script
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"face_landmarker.task not found at {model_path}. "
                "Download it by running: python download_model.py"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Canonical MediaPipe landmark indices
        # 6 points per eye for Soukupova & Cech EAR calculation
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        # 8 points for Mouth Aspect Ratio (MAR) yawn calculation
        # Outer corners [61, 291], upper lips [185, 12, 40], lower lips [84, 14, 314]
        self.MOUTH_IDX = [61, 185, 12, 40, 291, 314, 14, 84]

        # Key 3D facial reference model points for solvePnP head pose
        self.MODEL_POINTS_3D = np.array([
            (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
            (0.0, -330.0, -65.0),      # Chin (Landmark 152)
            (-225.0, 170.0, -135.0),   # Left eye left corner (Landmark 33)
            (225.0, 170.0, -135.0),    # Right eye right corner (Landmark 263)
            (-150.0, -150.0, -125.0),  # Left mouth corner (Landmark 61)
            (150.0, -150.0, -125.0)    # Right mouth corner (Landmark 291)
        ], dtype=np.float64)

        self.POSE_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

    def extract_landmarks(self, frame_rgb):
        """Processes RGB frame and returns normalized landmark coordinate list"""
        h, w, _ = frame_rgb.shape

        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None, None

        face_landmarks = result.face_landmarks[0]
        pixel_landmarks = []
        for lm in face_landmarks:
            pixel_landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))

        return face_landmarks, pixel_landmarks

    @staticmethod
    def calculate_ear(eye_points):
        """
        Computes Eye Aspect Ratio (EAR):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        p1, p2, p3, p4, p5, p6 = eye_points
        a = dist.euclidean(p2[:2], p6[:2])
        b = dist.euclidean(p3[:2], p5[:2])
        c = dist.euclidean(p1[:2], p4[:2])

        if c == 0:
            return 0.0
        ear = (a + b) / (2.0 * c)
        return float(ear)

    @staticmethod
    def calculate_mar(mouth_points):
        """
        Computes Mouth Aspect Ratio (MAR) for yawn detection:
        MAR = (||m2 - m8|| + ||m3 - m7|| + ||m4 - m6||) / (2 * ||m1 - m5||)
        """
        m1, m2, m3, m4, m5, m6, m7, m8 = mouth_points
        vertical_1 = dist.euclidean(m2[:2], m8[:2])
        vertical_2 = dist.euclidean(m3[:2], m7[:2])
        vertical_3 = dist.euclidean(m4[:2], m6[:2])
        horizontal = dist.euclidean(m1[:2], m5[:2])

        if horizontal == 0:
            return 0.0
        mar = (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
        return float(mar)

    def estimate_head_pose(self, pixel_landmarks, img_w, img_h):
        """
        Estimates Pitch, Yaw, Roll angles (in degrees) using cv2.solvePnP.
        - Pitch > 15 deg: Head nodding forward (drowsy head drop)
        - Yaw: Head turning left or right away from road
        - Roll: Head tilting to shoulder
        """
        image_points = np.array([
            pixel_landmarks[idx][:2] for idx in self.POSE_LANDMARK_IDX
        ], dtype=np.float64)

        # Camera intrinsic matrix approximation
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return pitch, yaw, roll

    def get_face_roi(self, frame, pixel_landmarks, target_size=(112, 112)):
        """Extracts and crops the normalized face bounding box for CNN feature extraction"""
        h, w, _ = frame.shape
        xs = [pt[0] for pt in pixel_landmarks]
        ys = [pt[1] for pt in pixel_landmarks]

        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))

        # Add 10% padding
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            return None, (0, 0, 0, 0)

        resized = cv2.resize(face_crop, target_size)
        return resized, (x_min, y_min, x_max - x_min, y_max - y_min)
