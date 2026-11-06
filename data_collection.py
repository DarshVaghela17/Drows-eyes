"""
Webcam Data Collection & Annotation Script
Allows recording and labeling custom driver datasets with 3 classes:
- Press 'a': Label active frame as 'Alert' (Normal driving posture)
- Press 'm': Label active frame as 'Mild Drowsiness' (Yawning / heavy eyes)
- Press 's': Label active frame as 'Severe Drowsiness' (Nodding off / eyes closed)
- Press 'q': Save and quit
Saves cropped facial ROI images to dataset/ and feature vectors to CSV.
"""

import cv2
import os
import csv
import time
from landmarks_detector import FacialLandmarkDetector

def setup_dataset_directories():
    base_dir = "dataset"
    classes = ["alert", "mild_drowsiness", "severe_drowsiness"]
    for c in classes:
        os.makedirs(os.path.join(base_dir, c), exist_ok=True)
    return base_dir

def main():
    base_dir = setup_dataset_directories()
    detector = FacialLandmarkDetector()
    cap = cv2.VideoCapture(0)

    csv_path = os.path.join(base_dir, "dataset_metadata.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "ear", "mar", "head_pitch", "head_yaw", "label"])

    print("=" * 60)
    print("DRIVER DROWSINESS DATA COLLECTION MODULE")
    print("Controls:")
    print("  [A] - Record as Alert")
    print("  [M] - Record as Mild Drowsiness (Yawn / Tired)")
    print("  [S] - Record as Severe Drowsiness (Closed Eyes / Head Nod)")
    print("  [Q] - Quit")
    print("=" * 60)

    sample_counts = {"alert": 0, "mild_drowsiness": 0, "severe_drowsiness": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, pixel_lm = detector.extract_landmarks(rgb)

        display_frame = frame.copy()
        ear = 0.3
        mar = 0.2
        pitch, yaw, roll = 0.0, 0.0, 0.0
        face_roi = None

        if pixel_lm:
            left_eye = [pixel_lm[i] for i in detector.LEFT_EYE_IDX]
            right_eye = [pixel_lm[i] for i in detector.RIGHT_EYE_IDX]
            mouth = [pixel_lm[i] for i in detector.MOUTH_IDX]

            ear = (detector.calculate_ear(left_eye) + detector.calculate_ear(right_eye)) / 2.0
            mar = detector.calculate_mar(mouth)
            pitch, yaw, roll = detector.estimate_head_pose(pixel_lm, w, h)
            face_roi, bbox = detector.get_face_roi(frame, pixel_lm)

            # Draw HUD
            bx, by, bw, bh = bbox
            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(display_frame, f"EAR: {ear:.3f} | MAR: {mar:.3f} | Pitch: {pitch:.1f}",
                        (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Instructions banner
        cv2.putText(display_frame, f"Alert: {sample_counts['alert']} | Mild: {sample_counts['mild_drowsiness']} | Severe: {sample_counts['severe_drowsiness']}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Driver Dataset Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        target_class = None
        if key == ord('a'):
            target_class = "alert"
        elif key == ord('m'):
            target_class = "mild_drowsiness"
        elif key == ord('s'):
            target_class = "severe_drowsiness"
        elif key == ord('q'):
            break

        if target_class and face_roi is not None:
            sample_counts[target_class] += 1
            filename = f"{target_class}_{int(time.time() * 1000)}.jpg"
            save_path = os.path.join(base_dir, target_class, filename)
            cv2.imwrite(save_path, face_roi)

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([save_path, f"{ear:.4f}", f"{mar:.4f}", f"{pitch:.2f}", f"{yaw:.2f}", target_class])
            print(f"Saved {target_class} frame #{sample_counts[target_class]} -> {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed successfully.")

if __name__ == '__main__':
    main()
