from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your AI model
model = tf.keras.models.load_model("urine_model.h5")  # Make sure you have the model file

# Define labels (modify based on your model)
LABELS = ["Normal", "Infection", "Kidney Disease", "Diabetes"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image file from request
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)[0]
        predicted_class = LABELS[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return jsonify({"result": f"{predicted_class} ({confidence}%)"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
