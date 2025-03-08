from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 🔹 Use absolute path for the model
MODEL_PATH = "C:/Users/Administrator/Downloads/keras_Model.h5"
LABELS_PATH = "C:/Users/Administrator/Downloads/labels.txt"

# ✅ Check if the model file exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    exit()

# ✅ Check if labels file exists
if not os.path.exists(LABELS_PATH):
    print(f"❌ Error: Labels file not found at {LABELS_PATH}")
    exit()

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Load class labels
class_names = open(LABELS_PATH, "r").readlines()

# Image processing function
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {image_path}")
        return

    # Load image and preprocess
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Prepare data for model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display results
    print(f"✅ Prediction: {class_name}")
    print(f"📊 Confidence Score: {confidence_score:.2f}")

# 🔹 Example usage (replace with actual image path)
IMAGE_PATH = "C:/Users/Administrator/Downloads/test_image.jpg"
predict_image(IMAGE_PATH)
