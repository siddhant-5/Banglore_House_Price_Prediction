import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open("model/banglore_hpp_v2.pickle", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")

st.title("🏠 Bangalore House Price Prediction")
st.markdown("Enter property details to estimate price")

# Inputs
location = st.text_input("Location")
sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0)
bath = st.number_input("Bathrooms", min_value=1, max_value=10)
bhk = st.number_input("BHK", min_value=1, max_value=10)

# Predict
if st.button("Predict Price"):
    try:
        input_data = np.array([[location, sqft, bath, bhk]])

        # If your model expects encoded input, adjust here
        prediction = model.predict(input_data)

        st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")

    except Exception as e:
        st.error(f"Error: {e}")