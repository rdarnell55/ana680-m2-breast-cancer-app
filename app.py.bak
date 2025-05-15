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

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        radius = float(request.form.get('radius', 0))
        texture = float(request.form.get('texture', 0))
        perimeter = float(request.form.get('perimeter', 0))
        area = float(request.form.get('area', 0))
        smoothness = float(request.form.get('smoothness', 0))

        logger.info(f"Received input: radius={radius}, texture={texture}, perimeter={perimeter}, area={area}, smoothness={smoothness}")

        data = [[radius, texture, perimeter, area, smoothness]]
        prediction_result = model.predict(data)[0]
        prediction = "Malignant" if prediction_result == 1 else "Benign"

        logger.info(f"Model prediction: {prediction}")
        return render_template('index.html',
                               prediction=prediction,
                               radius=radius,
                               texture=texture,
                               perimeter=perimeter,
                               area=area,
                               smoothness=smoothness)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)