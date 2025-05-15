#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_model.py
"""
Trains a K-Nearest Neighbors (KNN) model on selected features from the Breast Cancer Wisconsin (Original) dataset.
This script includes data preprocessing, scaling, model training, evaluation, and saving the trained model for deployment.
It is specifically configured for a simplified input of 5 features to align with a Flask web application form.

Steps:
1. Download the dataset directly from the UCI repository using `ucimlrepo`.
2. Select 5 relevant features to simplify user input.
3. Handle missing values with mean imputation.
4. Standardize feature values using `StandardScaler` to improve KNN performance.
5. Split the dataset into training and testing subsets.
6. Train a K-Nearest Neighbors model (k=5).
7. Evaluate the model's performance.
8. Save both the trained model and scaler into a `.pkl` file for use in the Flask app.

This is designed for use in a Flask CI/CD deployment pipeline.
"""

import pandas as pd  # Data manipulation
from sklearn.model_selection import train_test_split  # Splitting dataset into train/test
from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.preprocessing import StandardScaler  # Normalizing input data
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn.metrics import accuracy_score  # Evaluate model accuracy
from ucimlrepo import fetch_ucirepo  # Fetch UCI dataset
import pickle  # Save trained model and scaler to file

# === Step 1: Load the Breast Cancer dataset from UCI ===
print("Fetching dataset from UCI repository...")
dataset = fetch_ucirepo(id=15)  # ID 15 is the Breast Cancer Wisconsin (Original) dataset

# Combine features and target column into one DataFrame
print("Preparing dataset...")
data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
data.rename(columns={'Class': 'Target'}, inplace=True)  # Rename 'Class' to 'Target' for clarity

# Convert labels from (2 = benign, 4 = malignant) to (0 = benign, 1 = malignant)
data['Target'] = data['Target'].map({2: 0, 4: 1})

# === Step 2: Select 5 features to match the web form ===
# NOTE: The actual column names use underscores (_) instead of spaces.
selected_features = [
    'Clump_thickness',
    'Uniformity_of_cell_size',
    'Uniformity_of_cell_shape',
    'Marginal_adhesion',
    'Single_epithelial_cell_size'
]
X = data[selected_features]  # Feature set
y = data['Target']           # Target labels

# === Step 3: Handle missing values using mean imputation ===
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=selected_features)

# === Step 4: Scale features using StandardScaler ===
print("Scaling feature values...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === Step 5: Split data into training and testing sets ===
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# === Step 6: Train the KNN classifier (k=5) ===
print("Training K-Nearest Neighbors classifier...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# === Step 7: Evaluate model accuracy ===
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with 5 features: {accuracy:.4f}")

# === Step 8: Save trained model and scaler to file ===
print("Saving model and scaler to 'breast_cancer_model.pkl'...")
with open('breast_cancer_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

print("✅ Training complete. Model is ready for deployment.")


# In[ ]:




