import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Path to the train folder
DATADIR = "train"
CATEGORIES = ["Cat", "Dog"]
IMG_SIZE = 64

def load_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # 0 for Cat, 1 for Dog
        for img in tqdm(os.listdir(path), desc=f"Loading {category} images"):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_array is None:
                    continue  # skip unreadable images
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                continue
    return data

# Load and prepare data
print("[INFO] Loading dataset...")
data = load_data()
print(f"[INFO] Loaded {len(data)} images.")

# Shuffle and separate features/labels
np.random.shuffle(data)
X = np.array([features for features, label in data]).reshape(-1, IMG_SIZE * IMG_SIZE * 3)
y = np.array([label for features, label in data])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
print("[INFO] Training SVM...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n[RESULTS]")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))
