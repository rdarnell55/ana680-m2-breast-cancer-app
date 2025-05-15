#!/usr/bin/env python
# coding: utf-8

# In[3]:


# train_model.py
"""
This script trains a K-Nearest Neighbors (KNN) classifier to predict breast cancer malignancy
using the Breast Cancer Wisconsin (Original) dataset from the UCI repository. It handles data
cleaning, imputation, scaling, model training, evaluation, and finally saves all components
(model, imputer, scaler) for later use in deployment.
"""

# === Step 1: Import libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from ucimlrepo import fetch_ucirepo

# === Step 2: Load the dataset ===
dataset = fetch_ucirepo(id=15)  # Breast Cancer Wisconsin (Original)
data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
data.rename(columns={'Class': 'Target'}, inplace=True)
data['Target'] = data['Target'].map({2: 0, 4: 1})  # Benign: 0, Malignant: 1

# === Step 3: Separate features and labels ===
X = data.drop('Target', axis=1)
y = data['Target']

# === Step 4: Impute missing values ===
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# === Step 5: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.25, random_state=42
)

# === Step 6: Scale the features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 7: Train KNN model ===
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# === Step 8: Evaluate accuracy ===
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# === Step 9: Save the model ===
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as model.pkl")

# === Step 10: Save the imputer ===
with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
print("Imputer saved as imputer.pkl")

# === Step 11: Save the scaler ===
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved as scaler.pkl")


# In[ ]:




