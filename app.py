import streamlit as st
import pickle
import numpy as np

st.title("Bean Classifier App")

# Load the trained model
with open("bean_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler used during training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

features_list = [
    "area", "perimeter", "major_axis_length", "minor_axis_length", "aspect_ratio",
    "eccentricity", "convex_area", "equiv_diameter", "extent", "solidity",
    "roundness", "compactness", "shape_factor1", "shape_factor2", "shape_factor3", "shape_factor4"
]

st.write("Enter the values for the following features:")

input_data = []
for feature in features_list:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    X = np.array([input_data])
    X_scaled = scaler.transform(X)  # use the original scaler
    prediction = model.predict(X_scaled)
    st.success(f"Predicted Bean Class: {prediction[0]}")
