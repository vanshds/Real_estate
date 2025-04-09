import streamlit as st
import numpy as np
import pickle
import json

# Load the model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load columns
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Extract locations (after 3 features: sqft, bath, bhk)
locations = data_columns[3:]

# --- Streamlit UI ---
st.set_page_config(page_title="🏠 Bangalore House Price Predictor", layout="centered")
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Enter the details below to estimate the house price:")

# Input form
sqft = st.number_input("Total Square Feet", min_value=100, max_value=10000, value=1000)
bath = st.slider("Bathrooms", 1, 5, 2)
bhk = st.slider("Bedrooms (BHK)", 1, 5, 2)
location = st.selectbox("Location", sorted(locations))

# Predict function
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    try:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1
    except:
        pass
    return round(model.predict([x])[0], 2)

# Predict button
if st.button("Predict Price"):
    result = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {result:,.2f}")
