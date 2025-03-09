import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy endpoint for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running!"})

# Example prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Assuming input is JSON
    # Process the data with your model here
    result = {"prediction": "Sample result"}  # Replace with actual model output
    return jsonify(result)

# Get PORT from environment variables (Render requirement)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if not set
    app.run(host="0.0.0.0", port=port)
