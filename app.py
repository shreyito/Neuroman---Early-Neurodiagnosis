from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_fold_1.h5")
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return "NeuroMan Model API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_extension = file.filename.split(".")[-1]

    try:
        # Convert file content into a Pandas DataFrame
        if file_extension == "csv":
            df = pd.read_csv(file)
        elif file_extension == "json":
            df = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or JSON."}), 400

        input_array = np.array(df.values).reshape(1, -1)  # Adjust shape if needed

        # Make prediction
        prediction = model.predict(input_array)[0][0]  # Assuming a single output value

        # Check if Parkinson's is detected
        if prediction > 0.8:
            result = {
                "prediction": float(prediction),
                "message": "Possibility of Parkinson Detected"
            }
        else:
            result = {
                "prediction": float(prediction),
                "message": "Low Possibility of Parkinson"
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Change `debug=False` in production
