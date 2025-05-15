from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    with open("breast_cancer_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
        logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model and scaler: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract all 9 features from the form
        features = [
            'clump_thickness',
            'uniformity_cell_size',
            'uniformity_cell_shape',
            'marginal_adhesion',
            'single_epithelial_cell_size',
            'bare_nuclei',
            'bland_chromatin',
            'normal_nucleoli',
            'mitoses'
        ]
        
        # Retrieve and convert form inputs to float
        input_values = [float(request.form.get(f, 0)) for f in features]
        logger.info(f"📥 Received input: {input_values}")

        # Scale input
        input_scaled = scaler.transform([input_values])

        # Make prediction
        prediction_result = model.predict(input_scaled)[0]
        prediction = "Malignant" if prediction_result == 1 else "Benign"
        logger.info(f"🔮 Prediction: {prediction}")

        # Send inputs back for UI persistence
        return render_template('index.html', prediction=prediction, **dict(zip(features, input_values)))
    except Exception as e:
        logger.error(f"❗ Error during prediction: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)