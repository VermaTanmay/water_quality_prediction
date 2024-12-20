import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Create Streamlit app layout
st.title("Water Potability Prediction")

# Input fields for water quality parameters
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=0.1)
solids = st.number_input("Solids (mg/L)", min_value=0.0, value=20000.0, step=10.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0, step=10.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=500.0, step=10.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=70.0, step=0.1)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0, step=0.1)

# Create a DataFrame for the input features
input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                          columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"])

# Scale the input data using the pre-trained scaler
try:
    input_data_scaled = scaler.transform(input_data)
except ValueError as e:
    st.error(f"Feature mismatch error: {e}")
    st.stop()

# Button to make a prediction
if st.button("Predict Water Potability"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("The water is potable!")
    else:
        st.error("The water is not potable.")
