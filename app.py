from flask import Flask, render_template, request
import pickle
import pandas as pd
import logging
import os  # Needed for Heroku to get the PORT

# === Initialize the Flask web application ===
app = Flask(__name__)

# === Load the trained machine learning model ===
# Also load the imputer used during training to preprocess input data
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("imputer.pkl", "rb") as imputer_file:
        imputer = pickle.load(imputer_file)
except Exception as e:
    # If loading fails, the app shouldn't even try to run
    raise RuntimeError(f"Failed to load model or imputer: {e}")

# === Configure logging to help with debugging ===
logging.basicConfig(level=logging.INFO)

# === Home Route ===
# Renders the HTML form when user visits the homepage
@app.route("/")
def home():
    return render_template("index.html")

# === Prediction Route ===
# This runs when the user submits the form
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract and convert input features from form to floats
        input_data = pd.DataFrame([[
            float(request.form['Clump_thickness']),
            float(request.form['Uniformity_of_cell_size']),
            float(request.form['Uniformity_of_cell_shape']),
            float(request.form['Marginal_adhesion']),
            float(request.form['Single_epithelial_cell_size']),
            float(request.form['Bare_nuclei']),
            float(request.form['Bland_chromatin']),
            float(request.form['Normal_nucleoli']),
            float(request.form['Mitoses'])
        ]], columns=[
            'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape',
            'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei',
            'Bland_chromatin', 'Normal_nucleoli', 'Mitoses'
        ])

        # Log the raw input data for debugging
        logging.info("Raw input data received:\n%s", input_data)

        # Preprocess the input data using the same imputer from training
        input_imputed = imputer.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_imputed)[0]

        # Interpret result: 0 = Benign, 1 = Malignant
        prediction_text = "Malignant" if prediction == 1 else "Benign"

        # Log the prediction
        logging.info("Prediction result: %s", prediction_text)

        # Re-render the form with the result displayed
        return render_template("index.html", prediction=prediction_text, **request.form)

    except Exception as e:
        # Log any error that occurs during processing
        logging.error("Prediction failed: %s", e)
        return render_template("index.html", prediction=f"Error: {e}", **request.form)

# === Required block to run the app on Heroku ===
# Heroku assigns a dynamic port, which must be read from the environment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local testing
    app.run(host='0.0.0.0', port=port, debug=True)