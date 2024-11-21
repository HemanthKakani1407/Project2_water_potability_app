import streamlit as st
import pandas as pd
import numpy as np
import pickle

#loading the saved model and scaler
with open('knn_water_potability_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#webpage

st.title("Water Potability Prediction App")

st.write ("""This app uses a **k-Nearest Neighbors (k-NN)** machine learning model to predict water potability based on quality parameters like pH, hardness, and turbidity. 
Simply adjust the sliders and dropdowns to input water sample data and check if the water is potable!""")

st.sidebar.header("Input Water Quality Parameters")
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0, step=0.1)
hardness = st.sidebar.slider("Hardness", 0.0, 300.0, 150.0)
solids = st.sidebar.slider("Total Dissolved Solids", 0.0, 50000.0, 20000.0)
chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 15.0, 5.0, step=0.1)
sulfate = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0)
conductivity = st.sidebar.slider("Conductivity (μS/cm)", 0.0, 800.0, 400.0)
organic_carbon = st.sidebar.slider("Organic Carbon (mg/L)", 0.0, 30.0, 15.0, step=0.1)
trihalomethanes = st.sidebar.slider("Trihalomethanes (μg/L)", 0.0, 120.0, 60.0, step=0.1)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 5.0, step=0.1)

input_data = pd.DataFrame({
    'ph': [ph],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
})

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = model.predict_proba(input_data_scaled)

if prediction[0] == 1:
    st.success("The water is **Potable** 💧")
else:
    st.error("The water is **Not Potable** 🚫")

st.write(f"**Prediction Probability**: Potable: {prediction_proba[0][1]*100:.2f}%, Not Potable: {prediction_proba[0][0]*100:.2f}%")

st.subheader("Water Quality Parameters")
st.write("Here is the summary of the input water quality parameters:")
st.write(input_data)