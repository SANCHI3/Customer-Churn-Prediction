import streamlit as st
import joblib

# Load pipeline
pipeline = joblib.load("churn_pipeline.pkl")
model = pipeline["model"]

st.set_page_config(page_title="Customer Churn Predictor")

st.title("📊 Customer Churn Prediction System")
st.write("Predict telecom customer churn risk using Machine Learning")
st.write("VERSION 2")
# User Inputs
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

if st.button("Predict Churn"):
    st.success("Button Click Working!")
