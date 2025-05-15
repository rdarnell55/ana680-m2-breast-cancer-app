import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# === Load the trained model, imputer, and scaler ===
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("imputer.pkl", "rb") as imputer_file:
        imputer = pickle.load(imputer_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessors: {e}")

@app.route('/')
def home():
    # Display the input form with no prediction initially
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from the HTML form
        input_features = [
            float(request.form['Clump_thickness']),
            float(request.form['Uniformity_of_cell_size']),
            float(request.form['Uniformity_of_cell_shape']),
            float(request.form['Marginal_adhesion']),
            float(request.form['Single_epithelial_cell_size']),
            float(request.form['Bare_nuclei']),
            float(request.form['Bland_chromatin']),
            float(request.form['Normal_nucleoli']),
            float(request.form['Mitoses'])
        ]

        # Convert to 2D NumPy array
        input_data = np.array(input_features).reshape(1, -1)

        # Apply preprocessing: imputation followed by scaling
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)

        # Predict using the trained model
        prediction_numeric = model.predict(input_scaled)[0]
        prediction = 'Malignant' if prediction_numeric == 1 else 'Benign'

        # Render the result in the template
        return render_template(
            'index.html', 
            prediction=prediction,
            Clump_thickness=request.form['Clump_thickness'],
            Uniformity_of_cell_size=request.form['Uniformity_of_cell_size'],
            Uniformity_of_cell_shape=request.form['Uniformity_of_cell_shape'],
            Marginal_adhesion=request.form['Marginal_adhesion'],
            Single_epithelial_cell_size=request.form['Single_epithelial_cell_size'],
            Bare_nuclei=request.form['Bare_nuclei'],
            Bland_chromatin=request.form['Bland_chromatin'],
            Normal_nucleoli=request.form['Normal_nucleoli'],
            Mitoses=request.form['Mitoses']
        )

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# === Run the Flask app (bind to appropriate host/port for Heroku) ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)