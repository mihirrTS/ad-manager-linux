import cv2
import numpy as np
import os
import random
import time
import threading
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import subprocess
import shutil
import sys

# ======== Load Models ========
MODEL_DIR = "models"
gender_model = load_model(os.path.join(MODEL_DIR, "gender_model.keras"))
age_model = load_model(os.path.join(MODEL_DIR, "age_model.keras"))

GENDER_LABELS = ['female', 'male']
AGE_LABELS = ['adult', 'kid', 'teen']

CATEGORIES = [
    "male_kid", "male_teen", "male_adult",
    "female_kid", "female_teen", "female_adult"
]

ADS_FOLDER = "ads"
os.makedirs(ADS_FOLDER, exist_ok=True)

# ======== Global Configurable State ========
ad_queue = []
is_playing_ad = False
last_seen_time = 0
DETECTION_TIMEOUT = 5
ad_lock = threading.Lock()
video_play_lock = threading.Lock()  # 🔒 Ensures only 1 video plays at a time

# Defaults (can be overridden from command-line)
VIDEO_SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
PLAY_DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# ======== Face Detector ========
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def open_camera():
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    raise RuntimeError("❌ No webcam found.")

# ======== Play Video ========
def play_video(video_path, duration_limit=5, speed=1.0):
    global is_playing_ad

    if not shutil.which("mpv"):
        print("[ERROR] mpv is not installed. Install it with: sudo apt install mpv")
        return

    is_playing_ad = True
    try:
        subprocess.run([
            "mpv",
            "--fs",
            "--no-terminal",
            "--really-quiet",
            f"--length={duration_limit}",
            f"--speed={speed}",
            video_path
        ])
    except Exception as e:
        print(f"[MPV ERROR] {e}")
    is_playing_ad = False

def get_category_from_detection(face_img):
    try:
        face = cv2.resize(face_img, (160, 160))
        face = preprocess_input(face.astype("float32"))
        face = np.expand_dims(face, axis=0)
        gender_pred = gender_model.predict(face, verbose=0)
        age_pred = age_model.predict(face, verbose=0)
        gender = GENDER_LABELS[np.argmax(gender_pred)]
        age = AGE_LABELS[np.argmax(age_pred)]
        category = f"{gender}_{age}"
        print(f"[DETECTED] {gender}, {age} → Category: {category}")
        return category if category in CATEGORIES else "unknown"
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "unknown"

def detect_faces():
    global ad_queue, last_seen_time
    cap = open_camera()
    with face_detector as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if result.detections:
                largest_box = None
                max_area = 0
                for detection in result.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    bw = int(bboxC.width * w)
                    bh = int(bboxC.height * h)
                    area = bw * bh
                    if area > max_area:
                        max_area = area
                        largest_box = (x, y, bw, bh)
                if largest_box:
                    x, y, bw, bh = largest_box
                    face_img = frame[y:y+bh, x:x+bw]
                    if face_img.shape[0] >= 50 and face_img.shape[1] >= 50:
                        category = get_category_from_detection(face_img)
                        with ad_lock:
                            if category in CATEGORIES:
                                ad_queue = [category]
                                last_seen_time = time.time()
            time.sleep(1)

def main_loop():
    while True:
        with ad_lock:
            recent_detection = time.time() - last_seen_time < DETECTION_TIMEOUT
            current_queue = list(ad_queue)

        if recent_detection and current_queue:
            category = current_queue.pop(0)
            category_path = os.path.join(ADS_FOLDER, category)
            if os.path.exists(category_path):
                video_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.mp4', '.avi'))]
                if video_files:
                    video_path = os.path.join(category_path, random.choice(video_files))
                    print(f"[TARGETED] Playing: {video_path}")
                    with video_play_lock:  # 🔒 Ensure only one plays
                        play_video(video_path, duration_limit=PLAY_DURATION, speed=VIDEO_SPEED)
                    continue

        # Play random ad if nothing detected
        all_videos = []
        for cat in CATEGORIES:
            cat_path = os.path.join(ADS_FOLDER, cat)
            if os.path.exists(cat_path):
                for video in os.listdir(cat_path):
                    if video.lower().endswith(('.mp4', '.avi')):
                        all_videos.append(os.path.join(cat_path, video))
        if all_videos:
            print("[RANDOM] Playing random ad")
            with video_play_lock:  # 🔒
                play_video(random.choice(all_videos), duration_limit=PLAY_DURATION, speed=VIDEO_SPEED)

        time.sleep(0.1)

if __name__ == "__main__":
    print("📺 Starting Ad Display System... Press Ctrl+C to stop.")
    face_thread = threading.Thread(target=detect_faces, daemon=True)
    face_thread.start()
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[EXIT] Program stopped by user.")
        cv2.destroyAllWindows()
