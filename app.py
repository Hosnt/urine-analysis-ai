import os
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps

app = Flask(__name__)

# 🔹 Use dynamic paths for deployment
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

# ✅ Check if model & labels exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("❌ Model or labels file is missing!")

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Load class labels
class_names = open(LABELS_PATH, "r").readlines()

# 🔹 Function to preprocess & predict
def predict_image(image):
    image = Image.open(image).convert("RGB")
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
    confidence_score = float(prediction[0][index])

    return {"prediction": class_name, "confidence": confidence_score}

# 🔹 API Route for image upload & prediction
@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    result = predict_image(file)

    return jsonify(result)

# 🔹 Home route to check if API is running
@app.route("/")
def home():
    return "Urine Analysis AI API is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses port 10000
    app.run(host="0.0.0.0", port=port)
