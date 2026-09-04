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

# Load training data to dynamically populate dropdown options
@st.cache_data
def load_data():
    df = pd.read_csv('train-data.csv')
    df['Brand'] = df['Name'].apply(lambda x: str(x).split()[0])
    df['Model'] = df['Name'].apply(lambda x: str(x).split()[1] if len(str(x).split()) > 1 else '')
    return df

model = load_model()
df = load_data()

st.title("🚗 Car Price Prediction App")
st.write("Enter the vehicle specifications below to predict the estimated price.")

# 1. Location selection (Required by model feature pipeline)
locations = sorted(df['Location'].dropna().unique().tolist())
location = st.selectbox("Location", locations)

# 2. Dynamic Brand and Model selection
brands = sorted(df['Brand'].dropna().unique().tolist())
brand = st.selectbox("Brand", brands)

# Filter model choices dynamically based on selected Brand
available_models = sorted(df[df['Brand'] == brand]['Model'].dropna().unique().tolist())
car_model = st.selectbox("Model", available_models)

# 3. Additional Vehicle Specifications
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2015)
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

fuel_types = sorted(df['Fuel_Type'].dropna().unique().tolist())
fuel_type = st.selectbox("Fuel Type", fuel_types)

transmissions = sorted(df['Transmission'].dropna().unique().tolist())
transmission = st.selectbox("Transmission", transmissions)

owner_types = sorted(df['Owner_Type'].dropna().unique().tolist())
owner_type = st.selectbox("Owner Type", owner_types)

seats_list = [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
seats = st.selectbox("Number of Seats", seats_list)

engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=6000, value=1200)
power = st.number_input("Power (bhp)", min_value=30.0, max_value=600.0, value=85.0)
mileageklm = st.number_input("Mileage (km/l or km/kg)", min_value=0.0, value=20.0)

if st.button("Predict Price"):
    # Recreate feature DataFrame matching exact model expected input columns
    input_data = pd.DataFrame([{
        'Location': location,
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