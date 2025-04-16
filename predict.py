import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = 64
MODEL_PATH = "logistic_model.pkl"
IMAGE_PATH = "test7.jpg"  # Change this to your test image filename

# === Load the trained model ===
model = joblib.load(MODEL_PATH)

# === Preprocess the new image ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).flatten() / 255.0
    return img_array.reshape(1, -1)

# === Predict ===
x = preprocess_image(IMAGE_PATH)
prediction = model.predict(x)[0]
label = "Dog 🐶" if prediction == 1 else "Cat 🐱"

# === Show result ===
print(f"\n🎯 Prediction: {label}")

# Optional: Show the image too
img = Image.open(IMAGE_PATH)
plt.imshow(img)
plt.title(f"Prediction: {label}")
plt.axis('off')
plt.show()
