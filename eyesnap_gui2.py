import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import smtplib
from email.message import EmailMessage
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from plyer import notification

RIGHT_EYE_IDX = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_EYE_IDX  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
IRIS_RANGE = range(468, 478)

def landmarks_to_array(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

def centroid(points, idxs):
    pts = points[idxs]
    return pts.mean(axis=0)

def nearest_point(pts, ref_pt):
    d = np.linalg.norm(pts - ref_pt, axis=1)
    return pts[np.argmin(d)]

def eye_centered(iris_pt, eye_idxs, all_pts, tol=0.22):
    eye_pts = all_pts[eye_idxs]
    xmin = eye_pts[:,0].min()
    xmax = eye_pts[:,0].max()
    if xmax - xmin < 1e-6:
        return True
    norm_x = (iris_pt[0] - xmin) / (xmax - xmin)
    return abs(norm_x - 0.5) < tol

def send_email(img_path, to_email, from_email, app_password):
    try:
        msg = EmailMessage()
        msg['Subject'] = "EyeSnap Alert: Photo Captured"
        msg['From'] = from_email
        msg['To'] = to_email
        msg.set_content("A photo was captured because you looked away or no face was detected.")

        with open(img_path, 'rb') as f:
            img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(img_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print("Failed to send email:", e)

class EyeSnapApp(QWidget):
    # === Replace below with your details ===
    FROM_EMAIL = "your-email@gmail.com"
    APP_PASSWORD = "your-16-char-app-password"  # Gmail app password, NOT regular Gmail password
    TO_EMAIL = "recipient-email@gmail.com"
    SNAP_INTERVAL = 2.0  # seconds between snaps (and emails)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeSnap")

        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.last_snap = 0
        self.snap_interval = self.SNAP_INTERVAL
        self.tol = 0.22
        self.outdir = "samples"
        os.makedirs(self.outdir, exist_ok=True)
        self.looking = True

        self.image_label = QLabel()
        self.status_label = QLabel("Status: --")
        self.snap_count_label = QLabel("Snapshots taken: 0")
        self.snap_count = 0

        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.snap_count_label)
        layout.addWidget(self.btn_quit)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        left_iris_pt = None
        right_iris_pt = None
        eye_left_c = np.array([0,0], dtype=float)
        eye_right_c = np.array([0,0], dtype=float)
        seen_face = False
        looking = True

        if results.multi_face_landmarks:
            seen_face = True
            lm = results.multi_face_landmarks[0].landmark
            pts = landmarks_to_array(lm, w, h)

            eye_left_c = centroid(pts, LEFT_EYE_IDX)
            eye_right_c = centroid(pts, RIGHT_EYE_IDX)

            if len(pts) > max(IRIS_RANGE):
                iris_candidates = pts[IRIS_RANGE.start:IRIS_RANGE.stop]
                left_iris_pt  = nearest_point(iris_candidates, eye_left_c)
                right_iris_pt = nearest_point(iris_candidates, eye_right_c)

                left_centered = eye_centered(left_iris_pt, LEFT_EYE_IDX, pts, tol=self.tol)
                right_centered = eye_centered(right_iris_pt, RIGHT_EYE_IDX, pts, tol=self.tol)
                looking = left_centered and right_centered
            else:
                looking = True
        else:
            seen_face = False
            looking = False

        # Draw overlay
        if left_iris_pt is not None:
            cv2.circle(frame, tuple(left_iris_pt.astype(int)), 3, (0,255,0), -1)
        if right_iris_pt is not None:
            cv2.circle(frame, tuple(right_iris_pt.astype(int)), 3, (0,255,0), -1)
        cv2.circle(frame, tuple(eye_left_c.astype(int)), 2, (255,0,0), -1)
        cv2.circle(frame, tuple(eye_right_c.astype(int)), 2, (255,0,0), -1)
        status_text = "LOOKING" if looking else "AWAY"
        color = (0,200,0) if looking else (0,0,255)
        cv2.putText(frame, status_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        now = time.time()
        if (not looking or not seen_face) and (now - self.last_snap > self.snap_interval):
            fname = os.path.join(self.outdir, f"snap_{int(now)}.jpg")
            cv2.imwrite(fname, frame)
            self.last_snap = now
            self.snap_count += 1
            self.snap_count_label.setText(f"Snapshots taken: {self.snap_count}")

            notification.notify(
                title="EyeSnap",
                message=f"Captured photo: {fname}",
                timeout=3
            )
            # Send email
            send_email(fname, self.TO_EMAIL, self.FROM_EMAIL, self.APP_PASSWORD)

        # Show image in GUI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

        self.status_label.setText(f"Status: {status_text}")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = EyeSnapApp()
    window.show()
    sys.exit(app.exec())
