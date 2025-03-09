import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Urine Analysis AI is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Get assigned port from Render
    app.run(host="0.0.0.0", port=port)
