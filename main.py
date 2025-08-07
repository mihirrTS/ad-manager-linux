import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import numpy as np
import os
import random
import time
import threading
import mediapipe as mp
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import subprocess
import shutil
import sys
import unicodedata

# ========== CONFIG ==========
SHOW_OSD = 1  # 1 = show overlay label, 0 = no overlay
DEFAULT_SPEED = 1.0
DEFAULT_DURATION = 10

# ========== MODELS ==========
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

# ========== GLOBAL STATE ==========
ad_detections = []
last_seen_time = 0
DETECTION_TIMEOUT = 5  # sec
ad_lock = threading.Lock()
video_play_lock = threading.Lock()

# From command-line args or default
VIDEO_SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPEED
PLAY_DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DURATION

# ========== MEDIA PIPE ==========
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def open_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    raise RuntimeError("❌ No webcam found.")

# ========== PLAY VIDEO ==========
def play_video(video_path, duration_limit=5, speed=1.0, label=None):
    global is_playing_ad
    if not shutil.which("mpv"):
        print("[ERROR] mpv is not installed. Run: sudo apt install mpv")
        return

    is_playing_ad = True
    try:
        clean_label = unicodedata.normalize('NFKD', label or "").encode('ascii', 'ignore').decode('utf-8')
        clean_name = unicodedata.normalize('NFKD', os.path.basename(video_path)).encode('ascii', 'ignore').decode('utf-8')
        print(f"[INFO] Playing video: {clean_name}")

        cmd = [
            "mpv", "--fs", "--no-terminal", "--really-quiet",
            f"--speed={speed}",
            "--no-loop",  # Don't repeat
        ]

        if SHOW_OSD and clean_label:
            cmd += [
                "--osd-align-x=right",
                "--osd-align-y=bottom",
                "--osd-font-size=30",
                f"--osd-msg1={clean_label}"
            ]

        cmd.append(video_path)

        # Start the player
        proc = subprocess.Popen(cmd)
        start_time = time.time()

        # Monitor time
        while proc.poll() is None:
            elapsed = time.time() - start_time
            if elapsed >= duration_limit:
                proc.terminate()
                break
            time.sleep(0.1)

    except Exception as e:
        print(f"[MPV ERROR] {e}")
    is_playing_ad = False



# ========== PREDICT CATEGORY ==========
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
        label = f"{gender}, {age}"

        print(f"[DETECTION] {label} → Category: {category}")
        return (category if category in CATEGORIES else "unknown", label)
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return ("unknown", "")

# ========== DETECT FACES ==========
def detect_faces():
    global ad_detections, last_seen_time
    cap = open_camera()

    with face_detector as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            if result.detections:
                largest_box = None
                max_area = 0
                for detection in result.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    area = bw * bh
                    if area > max_area:
                        max_area = area
                        largest_box = (x, y, bw, bh)

                if largest_box:
                    x, y, bw, bh = largest_box
                    face_img = frame[y:y+bh, x:x+bw]
                    if face_img.shape[0] >= 50 and face_img.shape[1] >= 50:
                        category, label = get_category_from_detection(face_img)
                        with ad_lock:
                            if category in CATEGORIES:
                                ad_detections.append((category, label))
                                last_seen_time = time.time()
            time.sleep(0.5)  # ← 2 detections per second

# ========== MAIN LOOP ==========
def main_loop():
    global ad_detections
    print("[INFO] Presentation starting...")

    while True:
        with ad_lock:
            recent = time.time() - last_seen_time < DETECTION_TIMEOUT
            detections = list(ad_detections)
            ad_detections.clear()

        if recent and detections:
            category_counts = Counter([d[0] for d in detections])
            majority_category, majority_count = category_counts.most_common(1)[0]
            total = len(detections)

            print(f"[INFO] Top Detection: {majority_category} ({majority_count}/{total} detections)")

            # Get readable label like "male, teen" for the majority
            readable_label = next((lbl for cat, lbl in detections if cat == majority_category), majority_category)

            # Combine readable label + stats into multiline string
            osd_label = f"{readable_label}\n{majority_count}/{total} detections"

            category_path = os.path.join(ADS_FOLDER, majority_category)
            if os.path.exists(category_path):
                videos = [f for f in os.listdir(category_path) if f.lower().endswith(('.mp4', '.avi'))]
                if videos:
                    video_path = os.path.join(category_path, random.choice(videos))
                    print(f"[TARGETED] {os.path.basename(video_path)}")
                    with video_play_lock:
                        play_video(video_path, duration_limit=PLAY_DURATION, speed=VIDEO_SPEED, label=osd_label)
                    continue

        # fallback: random ad
        all_videos = []
        for cat in CATEGORIES:
            cat_path = os.path.join(ADS_FOLDER, cat)
            if os.path.exists(cat_path):
                for video in os.listdir(cat_path):
                    if video.lower().endswith(('.mp4', '.avi')):
                        all_videos.append(os.path.join(cat_path, video))

        if all_videos:
            video_path = random.choice(all_videos)
            print(f"[RANDOM] {os.path.basename(video_path)}")
            with video_play_lock:
                play_video(video_path, duration_limit=PLAY_DURATION, speed=VIDEO_SPEED)
        time.sleep(0.1)


# ========== RUN ==========
if __name__ == "__main__":
    print("📺 Starting Ad Display System...")
    face_thread = threading.Thread(target=detect_faces, daemon=True)
    face_thread.start()
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[EXIT] Program stopped by user.")
        cv2.destroyAllWindows()
