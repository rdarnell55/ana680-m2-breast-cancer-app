from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from pickle file
with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        features = [
            float(request.form['Clump Thickness']),
            float(request.form['Uniformity of Cell Size']),
            float(request.form['Uniformity of Cell Shape']),
            float(request.form['Marginal Adhesion']),
            float(request.form['Single Epithelial Cell Size']),
            float(request.form['Bare Nuclei']),
            float(request.form['Bland Chromatin']),
            float(request.form['Normal Nucleoli']),
            float(request.form['Mitotic Rate'])
        ]

        # Convert to 2D numpy array for model
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = 'Malignant' if prediction == 4 else 'Benign'

        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
