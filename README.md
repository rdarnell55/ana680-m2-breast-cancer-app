# 🧠 Breast Cancer Diagnosis App (CI/CD Deployment with Heroku)

## Project Overview

This project demonstrates a **complete machine learning pipeline** from model training to deployment. It uses the **Breast Cancer Wisconsin (Original)** dataset to build a binary classifier that predicts whether a tumor is **benign** or **malignant** based on cell features. The final application is deployed on **Heroku**, showcasing CI/CD workflows with GitHub Actions.

## Objective

The goal of this assignment was to:

- Train a machine learning model using scikit-learn.

- Deploy a Flask web app to Heroku.

- Automate deployment using GitHub Actions.

- Accept user input and return a real-time prediction.

## Tools and Technologies

- Python 3.10

- Flask (for web app)

- scikit-learn (KNN Classifier)

- Pandas, NumPy (data handling)

- Heroku (deployment)

- GitHub Actions (CI/CD pipeline)

## Model and Data

- **Model**: K-Nearest Neighbors (K=5)

- **Dataset**: [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

- **Features Used** (9 total):
  
  - Clump Thickness
  
  - Uniformity of Cell Size
  
  - Uniformity of Cell Shape
  
  - Marginal Adhesion
  
  - Single Epithelial Cell Size
  
  - Bare Nuclei
  
  - Bland Chromatin
  
  - Normal Nucleoli
  
  - Mitoses

- **Target Classes**:
  
  - `0` → Benign
  
  - `1` → Malignant

## App Structure

<pre>
```plaintext
project-root/
├── app.py # Flask web app
├── train_model.py # Training script
├── model.pkl # Trained model
├── imputer.pkl # Imputation transformer
├── scaler.pkl # Scaler transformer
├── requirements.txt # Dependencies
├── Procfile # Heroku process file
├── templates/
│ └── index.html # HTML form for the web app
├── .github/
│ └── workflows/
│ └── deploy.yml # GitHub Actions workflow
```
</pre>

## How It Works

1. User enters 9 cell features into a form on the web page.

2. The input is sent to the Flask backend via POST request.

3. Input is imputed and scaled using the same transformers used in training.

4. The model returns a binary prediction: **Benign** or **Malignant**.

5. Result is displayed directly on the web page.

## Testing

To verify functionality, several test cases were used—some skewed benign, others more suspicious. The final model demonstrated correct predictions based on known clinical patterns.

## Deployment

- App deployed to: [Heroku](https://rcd-mlapp-7d9ac813fc0f.herokuapp.com/)

- CI/CD: Any push to the `main` branch triggers deployment via GitHub Actions.

## Notes & Reflection

- Initially, only 5 features were used to simplify input. However, this resulted in low model accuracy.

- We updated the system to use **all 9 features** for better performance.

- The assignment emphasizes practical skills in building ML deployment pipelines, not just theoretical accuracy.