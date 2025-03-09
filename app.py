import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import tensorflowjs as tfjs

app = Flask(__name__)

# Load Teachable Machine Model from the 'model/' folder
MODEL_PATH = "./model/"
model = tfjs.converters.load_keras_model(MODEL_PATH + "model.json")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Process image for model input
    image = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)  # Get the class index

    result = "Normal" if class_idx == 0 else "Infected"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
