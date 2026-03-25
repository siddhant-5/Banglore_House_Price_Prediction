import streamlit as st
import pickle
import json
import numpy as np

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00c6ff;
}
.subtitle {
    font-size: 18px;
    color: #9aa0a6;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background: #161b22;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}
.stButton>button {
    background-color: #00c6ff;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_artifacts():
    with open("model/banglore_hpp_v2.pickle", "rb") as f:
        model = pickle.load(f)

    with open("model/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    return model, data_columns

model, data_columns = load_artifacts()

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# ---------------- HEADER ---------------- #
st.markdown('<div class="title">🏠 Bangalore House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart ML-powered real estate pricing</div>', unsafe_allow_html=True)

st.write("")

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([1, 1])

# -------- LEFT: INPUT -------- #
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📍 Property Details")

    location = st.selectbox("Location", data_columns[3:])

    sqft = st.number_input("Total Square Feet", 300.0, 10000.0, step=50.0)

    bath = st.slider("Bathrooms", 1, 10, 2)

    bhk = st.slider("BHK", 1, 10, 2)

    predict_btn = st.button("Predict Price")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- RIGHT: OUTPUT -------- #
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 💰 Prediction Result")

    if predict_btn:
        try:
            price = predict_price(location, sqft, bath, bhk)

            st.success(f"Estimated Price: ₹ {round(price, 2)} Lakhs")

            st.metric("Price Range", f"₹ {round(price, 2)} L")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Enter details and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

