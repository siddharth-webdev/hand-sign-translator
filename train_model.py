import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# Containers for features (X) and labels (y)
X = []
y = []

dataset_path = "dataset"

# --- Data Loading ---
# Iterate through each folder in the dataset directory (each folder is a label/sign)
for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    # Load every .npy file inside the folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if not os.path.isfile(file_path) or not file.endswith('.npy'):
            continue

        # Append the landmark data to X and the folder name to y
        data = np.load(file_path)
        X.append(data)
        y.append(label)

# Convert lists to NumPy arrays for Scikit-Learn
X = np.array(X)
y = np.array(y)

# --- Model Training ---
# Use a Random Forest with 200 decision trees
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# --- Export ---
# Save the trained model to a file so it can be used for real-time inference
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/asl_model.pkl")

print("✅ Model trained and saved")
