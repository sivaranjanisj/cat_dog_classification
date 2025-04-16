import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==== CONFIGURATION ====
CAT_DIR = 'CAT'  # folder with 6500 cat images
DOG_DIR = 'DOG'  # folder with 6500 dog images
IMG_SIZE = 64     # Resize to 64x64
MODEL_NAME = 'logistic_model.pkl'

# ==== Load and Preprocess Images ====
def load_images(folder, label):
    data = []
    count = 0
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            try:
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).convert('L')  # grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img).flatten() / 255.0  # normalize
                data.append((img_array, label))
                count += 1
                if count % 1000 == 0:
                    print(f"Loaded {count} images from {folder}...")
            except:
                continue
    return data

print("Loading images...")
cat_data = load_images(CAT_DIR, 0)
dog_data = load_images(DOG_DIR, 1)

print("Combining and shuffling data...")
all_data = cat_data + dog_data
np.random.shuffle(all_data)

X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])

# ==== Save Preprocessed Data (Optional) ====
np.save('X.npy', X)
np.save('y.npy', y)

# ==== Train Logistic Regression ====
print("Training Logistic Regression model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==== Evaluate ====
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# ==== Save Model ====
joblib.dump(model, MODEL_NAME)
print(f"\n🎉 Model saved as {MODEL_NAME}")
