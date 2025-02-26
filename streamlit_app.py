import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# Load trained model
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Load saved encoders and scalers
with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("poly.pkl", "rb") as poly_file:
    poly = pickle.load(poly_file)

# Streamlit UI Styling
st.set_page_config(page_title="Steel YS Predictor", page_icon="⚙️", layout="centered")
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; }
        h1 { color: #2E86C1; text-align: center; }
        .stButton>button { background-color: #2E86C1; color: white; font-size: 16px; padding: 10px; border-radius: 5px; }
        .stSuccess { font-size: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("⚙️ Steel Yield Strength Prediction")
st.write("Enter the input values below to predict the Yield Strength (YS) of Copper added steel.")

# User input with side-by-side layout
col1, col2 = st.columns(2)
cu_conc = col1.number_input("🔹 Cu Concentration (%)", value=0.0, step=0.01, format="%.2f")
ce = col2.number_input("🔹 CE Value", value=0.0, step=0.01, format="%.2f")
major_phase = st.selectbox("🔹 Major Phase", [
    "Martensitic", "Austenitic", "Martensitic+Austenitic",
    "Ferritic+Pearlitic", "Ferritic+Martensitic", "Ferritic"
])

# Prediction button with styling
if st.button("🔍 Predict Yield Strength"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Cu conc.': [cu_conc],
        ' CE': [ce],
        'Major Phase': [major_phase]
    })
    
    # One-hot encoding
    encoded_phase = encoder.transform(input_data[['Major Phase']])
    encoded_phase_df = pd.DataFrame(encoded_phase, columns=encoder.get_feature_names_out(['Major Phase']))
    
    # Concatenate numerical and encoded categorical features
    input_data = pd.concat([input_data.drop('Major Phase', axis=1).reset_index(drop=True), encoded_phase_df.reset_index(drop=True)], axis=1)
    
    # Standardize numerical features
    input_data[['Cu conc.', ' CE']] = scaler.transform(input_data[['Cu conc.', ' CE']])
    
    # Apply polynomial features
    input_data_poly = poly.transform(input_data)
    
    # Predict
    prediction = rf_model.predict(input_data_poly)
    
    # Display result with formatting
    st.success(f"✅ Predicted Yield Strength (YS): {prediction[0]:.2f} MPa")
