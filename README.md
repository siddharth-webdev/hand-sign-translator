# hand-sign-translator

This README file provides an overview and setup instructions for your sign language translation project. It covers the data collection, training, and real-time inference components of the system.

***

# AI Sign Language Translator

This project uses Google MediaPipe's Hand Landmarker and a Scikit-Learn Random Forest Classifier to translate American Sign Language (ASL) hand signs into text and speech in real-time.

## Project Structure

* **`collect_data.py`**: A script to capture hand landmark data (21 points in 3D space) using a webcam and save them as NumPy arrays.
* **`train_model.py`**: A script that loads the saved landmark data and trains a Random Forest Classifier to recognize specific signs.
* **`realtime_translate.py`**: The main application that processes live webcam feed, predicts signs using the trained model, builds sentences, and uses Text-to-Speech (TTS) to vocalize the results.
* **`hand_landmarker.task`**: The pre-trained MediaPipe hand landmark detection model.

## Requirements

To run this project, you will need:
* Python 3.x
* OpenCV (`cv2`)
* MediaPipe
* NumPy
* Scikit-Learn
* Joblib
* pyttsx3

Install dependencies via pip:
```bash
pip install opencv-python mediapipe numpy scikit-learn joblib pyttsx3
```

## How to Use

### 1. Data Collection
Run the collection script to build your dataset. You will be prompted to enter a label (e.g., "A", "B", "Hello"). 
```bash
python collect_data.py
```
* The script captures the (x, y, z) coordinates of 21 hand landmarks for every frame where a hand is detected.
* Data is saved in `dataset/[label]/[count].npy`.
* Press **ESC** to stop collecting for a specific label.

### 2. Training the Model
Once you have collected data for all desired signs, run the training script:
```bash
python train_model.py
```
* This script iterates through the `dataset` folder, loads all `.npy` files, and trains a `RandomForestClassifier`.
* The final model is saved as `model/asl_model.pkl`.

### 3. Real-time Translation
Run the translation script to start the application:
```bash
python realtime_translate.py
```
* **Inference**: The system uses a smoothing buffer of the last 15 predictions to ensure stability.
* **Sentence Building**: When a sign is consistently detected, it is added to the current sentence.
* **Text-to-Speech**: The app uses `pyttsx3` to speak each detected sign aloud.
* **Controls**:
    * **C**: Clear the current sentence and buffer.
    * **ESC**: Quit the application.

## Technical Details

* **Feature Extraction**: The model relies on 63 features per frame (21 landmarks × 3 coordinates) provided by MediaPipe.
* **Classification**: A Random Forest with 200 estimators is used for robust multi-class classification.
* **Smoothing**: A `deque` and `Counter` logic are implemented in the real-time script to prevent rapid "flickering" between predictions.
