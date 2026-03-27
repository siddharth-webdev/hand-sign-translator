import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

X = []
y = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if not os.path.isfile(file_path) or not file.endswith('.npy'):
            continue
        data = np.load(file_path)
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/asl_model.pkl")

print("✅ Model trained and saved")