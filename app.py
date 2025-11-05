import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and feature information
model = joblib.load('models/oil_sales_model.pkl')
feature_info = joblib.load('models/feature_info.pkl')

# Get feature names and numeric features
feature_names = feature_info['feature_names']
numeric_features = feature_info['numeric_features']

def prepare_input(input_data):
    # Create a DataFrame with all possible features initialized to 0
    input_encoded = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in the numeric features
    for feature in numeric_features:
        if feature != 'volume_sales':  # Skip volume_sales as it's derived
            input_encoded[feature] = input_data[feature]
    
    # Fill in the categorical features
    # City
    city_col = f"city_{input_data['city']}"
    if city_col in feature_names:
        input_encoded[city_col] = 1
    
    # Brand
    brand_col = f"brand_{input_data['brand']}"
    if brand_col in feature_names:
        input_encoded[brand_col] = 1
        
    # Class
    class_col = f"class_{input_data['oil_class']}"
    if class_col in feature_names:
        input_encoded[class_col] = 1
        
    # Manufacturer
    manufacturer_col = f"manufacturer_{input_data['manufacturer']}"
    if manufacturer_col in feature_names:
        input_encoded[manufacturer_col] = 1
        
    # Price bracket
    price_bracket_col = f"price_bracket_{input_data['price_bracket']}"
    if price_bracket_col in feature_names:
        input_encoded[price_bracket_col] = 1

    # Calculate volume_sales based on total value and average price
    input_encoded['volume_sales'] = 1.0  # Default to 1 unit
        
    return input_encoded

# Streamlit UI
st.title('Oil Sales Value Prediction')

# Input form
with st.form("prediction_form"):
    # Numeric inputs
    st.subheader("Numeric Inputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        average_price = st.number_input("Average Price (SAR)", min_value=0.0, value=50.0)
    with col2:
        year = st.number_input("Year", min_value=2022, max_value=2024, value=2024)
    with col3:
        month = st.number_input("Month", min_value=1, max_value=12, value=1)

    # Categorical inputs
    st.subheader("Categorical Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox("City", ['AL AHSA', 'AL BAHA', 'AL KHARJ', 'DAMMAM', 'HAIL', 
                                   'JAZAN', 'JEDDAH', 'MAKKAH', 'RIYADH', 'TABUK', 'TAIF', 'YANBU'])
        brand = st.selectbox("Brand", ['BAYTNA', 'GULF GOLD', 'HILAL', 'LARA', 'NAJMA', 
                                     'NOUR', 'RAWABI', 'RIMAL', 'SABAYA', 'ZAHRA'])
        
    with col2:
        manufacturer = st.selectbox("Manufacturer", ['AL HILAL INDUSTRIES', 'ARABIAN HARVEST CO', 
                                                   'BLUE OASIS CO', 'DESERT SUN CO', 'NAJDI CONSUMER', 
                                                   'NOVA FOODS', 'PALM & GRAIN GROUP', 'SAHARA EDIBLES'])
        oil_class = st.selectbox("Oil Class", ['CANOLA', 'COCONUT', 'CORN', 'SUNFLOWER', 'VEGETABLE'])

    # Price bracket selection
    price_brackets = ['11-20$', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', 
                     '81-90', '91-100', '101+']
    price_bracket = st.selectbox("Price Bracket", price_brackets)

    submit_button = st.form_submit_button("Predict Sales Value")

if submit_button:
    # Prepare input data
    input_data = {
        'average_price': average_price,
        'year': year,
        'month': month,
        'city': city,
        'brand': brand,
        'manufacturer': manufacturer,
        'oil_class': oil_class,
        'price_bracket': price_bracket
    }
    
    # Transform input
    input_encoded = prepare_input(input_data)
    
    # Make prediction
    prediction = model.predict(input_encoded)
    
    # Display results
    st.header("Prediction Results")
    st.write(f"Predicted Sales Value: SAR {prediction[0]:,.2f}")
    
    # Additional insights
    st.subheader("Input Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Product Details:")
        st.write(f"- Brand: {brand}")
        st.write(f"- Class: {oil_class}")
        st.write(f"- Price Bracket: {price_bracket}")
        
    with col2:
        st.write("Sale Details:")
        st.write(f"- Location: {city}")
        st.write(f"- Average Price: SAR {average_price:,.2f}")
        st.write(f"- Date: {month}/{year}")