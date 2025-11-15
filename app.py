# app.py
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title("Bean Classifier App")

# Load the saved model and scaler (if you saved scaler; if not, fit new scaler)
with open("bean_model.pkl", "rb") as f:
    model = pickle.load(f)

# List of features
features_list = [
    "area", "perimeter", "major_axis_length", "minor_axis_length", "aspect_ratio",
    "eccentricity", "convex_area", "equiv_diameter", "extent", "solidity",
    "roundness", "compactness", "shape_factor1", "shape_factor2", "shape_factor3", "shape_factor4"
]

st.write("Enter the values for the following features:")

# Collect user input for all features
input_data = []
for feature in features_list:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    X = np.array([input_data])

    # Optional: scale features if your model was trained on scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # ideally, you should save & load original scaler
    prediction = model.predict(X_scaled)

    st.success(f"Predicted Bean Class: {prediction[0]}")
