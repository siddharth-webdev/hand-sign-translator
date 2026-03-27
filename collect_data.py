import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import time

label = input("Enter label (e.g., A): ")
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

count = 0

def save_result(result, output_image, timestamp_ms):
    global count
    if result.hand_landmarks:
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])

        np.save(f"{save_dir}/{count}.npy", np.array(landmarks))
        count += 1

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=save_result)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time() * 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000 - start_time)
        landmarker.detect_async(mp_image, timestamp_ms)
        frame_count += 1

        cv2.putText(frame, f"Samples: {count}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()