import cv2
import numpy as np
import os
import random
import time
import threading
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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
os.makedirs(ADS_FOLDER, exist_ok=True)  # Ensure ads/ exists

# ======== Global State ========
ad_queue = []
is_playing_ad = False
last_seen_time = 0
DETECTION_TIMEOUT = 5
ad_lock = threading.Lock()

# ======== Face Detector (MediaPipe) ========
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# ======== Webcam =========
def open_camera():
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    raise RuntimeError("❌ No webcam found.")

# ======== Play Video with Label Overlay ========
def play_video(video_path, duration_limit=5):
    global is_playing_ad, last_prediction_label
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    cv2.namedWindow("Ad Display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Ad Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_playing_ad = True
    start_time = time.time()

    while cap.isOpened() and (time.time() - start_time < duration_limit):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow("Ad Display", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Ad Display")
    is_playing_ad = False

# ======== Predict Category from Face Image ========
def get_category_from_detection(face_img):
    global last_prediction_label
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
        last_prediction_label = label
        print(f"[DETECTED] {label} → Category: {category}")

        return category if category in CATEGORIES else "unknown"
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "unknown"

# ======== Detect Faces and Add to Queue ========
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
                # Pick the largest face
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

# ======== Main Loop =========
def main_loop():
    global ad_queue, is_playing_ad

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
                    play_video(video_path)
                    continue

        # Otherwise play random ad
        all_videos = []
        for cat in CATEGORIES:
            cat_path = os.path.join(ADS_FOLDER, cat)
            if os.path.exists(cat_path):
                for video in os.listdir(cat_path):
                    if video.lower().endswith(('.mp4', '.avi')):
                        all_videos.append(os.path.join(cat_path, video))

        if all_videos:
            print("[RANDOM] Playing random ad")
            play_video(random.choice(all_videos))

        time.sleep(0.1)

# ======== Run ========
if __name__ == "__main__":
    print("📺 Starting Ad Display System... Press Ctrl+C to stop.")
    face_thread = threading.Thread(target=detect_faces)
    face_thread.daemon = True
    face_thread.start()

    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[EXIT] Program stopped by user.")
        cv2.destroyAllWindows()
