import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import pyttsx3

# -------- Load model --------
model = joblib.load("model/asl_model.pkl")

# -------- TTS --------
engine = pyttsx3.init()

# -------- MediaPipe Tasks API --------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------- Smoothing buffer --------
buffer = deque(maxlen=15)
sentence = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_count * 33)
        frame_count += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        prediction = ""

        if result and result.hand_landmarks:
            # first hand only
            hand_landmarks = result.hand_landmarks[0]

            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                pred = model.predict([landmarks])[0]
                buffer.append(pred)

                if len(buffer) == buffer.maxlen:
                    prediction = Counter(buffer).most_common(1)[0][0]
            else:
                prediction = ""

            # optionally draw simple landmarks points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        # -------- Sentence builder --------
        if prediction:
            if not sentence or prediction != sentence[-1]:
                sentence += prediction
                engine.say(prediction)
                engine.runAndWait()

        # -------- Display --------
        cv2.putText(frame, f"Current: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Press ESC to quit, C to clear", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Production Sign Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("c"):
            sentence = ""
            buffer.clear()

cap.release()
cv2.destroyAllWindows()