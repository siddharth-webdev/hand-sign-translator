import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import pyttsx3

# -------- Load model --------
# Load the pre-trained classifier
model = joblib.load("model/asl_model.pkl")

# -------- TTS --------
# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# -------- MediaPipe Tasks API --------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure landmarker for VIDEO mode (synchronous processing per frame)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------- Smoothing buffer --------
# Use a deque to store the last 15 predictions to reduce flickering/jitter
buffer = deque(maxlen=15)
sentence = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0

# --- Inference Loop ---
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # In VIDEO mode, timestamps must be monotonically increasing
        timestamp_ms = int(frame_count * 33)
        frame_count += 1

        # Process the image to find landmarks
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        prediction = ""

        if result and result.hand_landmarks:
            # Extract landmarks for the first hand detected
            hand_landmarks = result.hand_landmarks[0]

            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Ensure we have all 63 coordinates (21 points * 3 dimensions)
            if len(landmarks) == 63:
                # Use the ML model to predict the label
                pred = model.predict([landmarks])[0]
                buffer.append(pred)

                # Use the most frequent prediction in the buffer to smooth the result
                if len(buffer) == buffer.maxlen:
                    prediction = Counter(buffer).most_common(1)[0][0]
            else:
                prediction = ""

            # Draw circles on the landmarks for visual feedback
            for lm in hand_landmarks:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        # -------- Sentence builder --------
        # If a stable prediction exists and it differs from the last character in the sentence
        if prediction:
            if not sentence or prediction != sentence[-1]:
                sentence += prediction
                # Speak the new character
                engine.say(prediction)
                engine.runAndWait()

        # -------- UI Display --------
        cv2.putText(frame, f"Current: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Press ESC to quit, C to clear", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Production Sign Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        if key == ord("c"): # Clear sentence and buffer
            sentence = ""
            buffer.clear()

cap.release()
cv2.destroyAllWindows()
