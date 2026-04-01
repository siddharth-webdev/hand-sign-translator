import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import time

# --- Setup and Initialization ---
# Get the label name from the user to create a specific folder for that sign
label = input("Enter label (e.g., A): ")
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

# Path to the MediaPipe hand landmarker model file
model_path = 'hand_landmarker.task'

# MediaPipe Task aliases for easier access
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

count = 0 # Counter for saved data samples

# --- Callback Function ---
# This function is called by MediaPipe every time it processes a frame
def save_result(result, output_image, timestamp_ms):
    global count
    # Check if any hands were detected in the frame
    if result.hand_landmarks:
        landmarks = []
        # Extract x, y, z coordinates for all 21 hand landmarks (first hand only)
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Save the landmark array (63 values) as a NumPy file
        np.save(f"{save_dir}/{count}.npy", np.array(landmarks))
        count += 1

# --- MediaPipe Configuration ---
# Set up the landmarker for Live Stream mode which uses a callback
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=save_result)

# --- Main Capture Loop ---
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time() * 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calculate timestamp for the live stream processor
        timestamp_ms = int(time.time() * 1000 - start_time)
        # Send the frame to the landmarker; results handled in save_result()
        landmarker.detect_async(mp_image, timestamp_ms)
        frame_count += 1

        # Display the number of samples collected on screen
        cv2.putText(frame, f"Samples: {count}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collecting", frame)
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
