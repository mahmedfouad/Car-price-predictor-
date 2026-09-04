# Deploying the model using Streamlit to be used as a web application for price prediction. The app will take user inputs for the features used in the model and return the predicted price.
# 1. write Price Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('linear_regression_model.pkl')

model = load_model()

st.title("🚗 Car Price Prediction App")
st.write("Enter the vehicle specifications below to predict the estimated price.")

# Form inputs
brand = st.text_input("Brand", "Maruti")
car_model = st.text_input("Model", "Wagon R")
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2015)
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel_type = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
seats = st.selectbox("Number of Seats", [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=6000, value=1200)
power = st.number_input("Power (bhp)", min_value=30.0, max_value=600.0, value=85.0)
mileageklm = st.number_input("Mileage (km/l or adjusted km/kg)", min_value=0.0, value=20.0)

if st.button("Predict Price"):
    # Recreate feature row matching model training input structure
    input_data = pd.DataFrame([{
        'Year': year,
        'Kilometers_Driven': kilometers_driven,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Owner_Type': owner_type,
        'Engine': engine,
        'Power': power,
        'Seats': seats,
        'Brand': brand,
        'Model': car_model,
        'Mileageklm': mileageklm,
        'Age': 2026 - year
    }])
    
    # Predict
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: **{prediction:.2f} Lakhs**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        