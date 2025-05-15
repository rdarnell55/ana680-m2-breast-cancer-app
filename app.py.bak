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
        # Get user input from the form
        clump_thickness = float(request.form.get('clump_thickness', 0))
        uniformity_cell_size = float(request.form.get('uniformity_cell_size', 0))
        uniformity_cell_shape = float(request.form.get('uniformity_cell_shape', 0))
        marginal_adhesion = float(request.form.get('marginal_adhesion', 0))
        epithelial_cell_size = float(request.form.get('epithelial_cell_size', 0))

        # Log the received values
        logger.info(f"📥 Received input: {clump_thickness}, {uniformity_cell_size}, {uniformity_cell_shape}, {marginal_adhesion}, {epithelial_cell_size}")

        # Prepare and scale input data
        input_data = np.array([[clump_thickness, uniformity_cell_size, uniformity_cell_shape, marginal_adhesion, epithelial_cell_size]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction_result = model.predict(input_scaled)[0]
        prediction = "Malignant" if prediction_result == 1 else "Benign"

        logger.info(f"🔮 Prediction: {prediction}")

        return render_template('index.html',
                               prediction=prediction,
                               clump_thickness=clump_thickness,
                               uniformity_cell_size=uniformity_cell_size,
                               uniformity_cell_shape=uniformity_cell_shape,
                               marginal_adhesion=marginal_adhesion,
                               epithelial_cell_size=epithelial_cell_size)
    except Exception as e:
        logger.error(f"❗ Error during prediction: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)