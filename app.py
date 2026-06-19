import streamlit as st
import joblib

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict telecom customer churn risk using Machine Learning")

pipeline = joblib.load("churn_pipeline.pkl")

st.success("Model Loaded Successfully!")
