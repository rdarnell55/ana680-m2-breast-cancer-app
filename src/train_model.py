#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_model.py

"""
This script trains a K-Nearest Neighbors (KNN) classifier on the Breast Cancer Wisconsin dataset.
It uses all 9 numeric features provided in the original dataset to predict whether a tumor is malignant or benign.

Steps:
1. Load dataset using ucimlrepo
2. Preprocess:
    - Drop ID columns if present
    - Handle missing values
    - Convert labels to binary (0 = benign, 1 = malignant)
3. Split the dataset into training and test sets
4. Train a KNN classifier
5. Evaluate model accuracy
6. Save the trained model as 'model.pkl'

This model will later be deployed using a Flask web app on Heroku.
"""

# === Imports ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from ucimlrepo import fetch_ucirepo

# === Step 1: Load the dataset ===
dataset = fetch_ucirepo(id=15)  # Breast Cancer Wisconsin (Original)
data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# Rename target column to 'Target' and map labels (2 = benign, 4 = malignant) to (0, 1)
data.rename(columns={'Class': 'Target'}, inplace=True)
data['Target'] = data['Target'].map({2: 0, 4: 1})

# === Step 2: Select all 9 features ===
feature_columns = [
    'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape',
    'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei',
    'Bland_chromatin', 'Normal_nucleoli', 'Mitoses'
]
X = data[feature_columns]
y = data['Target']

# === Step 3: Handle missing values ===
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)

# === Step 4: Split the data ===
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

# === Step 5: Train KNN classifier ===
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# === Step 6: Evaluate and save ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.4f}")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")


# In[ ]:




