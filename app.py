import os
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model from the correct directory
model_path = os.path.join(os.getcwd(), "model", "keras_Model.h5")
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at: {model_path}")

@app.route('/')
def home():
    return "Urine Analysis AI is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        image = Image.open(file).convert("RGB").resize((224, 224))  # Resize for model
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result = prediction.tolist()  # Convert to list for JSON response

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
