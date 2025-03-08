import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Ensure TensorFlow uses CPU (important for Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Load AI model (make sure 'urine_model.h5' is in the correct directory)
try:
    model = tf.keras.models.load_model("urine_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Avoid crashing if the model is missing

# Define labels based on your model's output classes
LABELS = ["Normal", "Infection", "Kidney Disease", "Diabetes"]

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "AI model not loaded"}), 500

    try:
        # Get image from request
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Preprocess image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # Resize for model
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)[0]
        predicted_class = LABELS[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return jsonify({"result": f"{predicted_class} ({confidence}%)"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
