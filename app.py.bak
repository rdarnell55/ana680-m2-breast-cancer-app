from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure model.pkl is in the root directory)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None  # Placeholder if the model fails to load

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded properly", 500

    try:
        features = [float(x) for x in request.form.values()]
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        result = "Malignant" if prediction == 1 else "Benign"
        return render_template('index.html', prediction_text=f'Tumor is likely: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)