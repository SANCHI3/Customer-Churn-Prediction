import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("churn_pipeline.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
feature_names = pipeline["feature_names"]

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict telecom customer churn risk using Machine Learning")

# ----------------------------
# CUSTOMER INPUTS
# ----------------------------

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.number_input("Tenure (Months)", 0, 72, 12)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

phone = st.selectbox("Phone Service", ["No", "Yes"])

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

multiple = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# ----------------------------
# PREDICTION
# ----------------------------

if st.button("Predict Churn"):

    row = {col: 0 for col in feature_names}

    # Binary columns
    row["gender"] = 1 if gender == "Male" else 0
    row["SeniorCitizen"] = senior
    row["Partner"] = 1 if partner == "Yes" else 0
    row["Dependents"] = 1 if dependents == "Yes" else 0
    row["PhoneService"] = 1 if phone == "Yes" else 0
    row["PaperlessBilling"] = 1 if paperless == "Yes" else 0

    # Numeric columns
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly
    row["TotalCharges"] = total

    # Feature Engineering
    row["avg_monthly_spend"] = total / (tenure + 1)

    services = [
        online_security == "Yes",
        online_backup == "Yes",
        device_protection == "Yes",
        tech_support == "Yes",
        streaming_tv == "Yes",
        streaming_movies == "Yes"
    ]

    row["num_services"] = sum(services)

    # Multiple Lines
    if multiple == "Yes":
        row["MultipleLines_Yes"] = 1
    elif multiple == "No phone service":
        row["MultipleLines_No phone service"] = 1

    # Internet
    if internet == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif internet == "No":
        row["InternetService_No"] = 1

    # Online Security
    if online_security == "Yes":
        row["OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        row["OnlineSecurity_No internet service"] = 1

    # Online Backup
    if online_backup == "Yes":
        row["OnlineBackup_Yes"] = 1
    elif online_backup == "No internet service":
        row["OnlineBackup_No internet service"] = 1

    # Device Protection
    if device_protection == "Yes":
        row["DeviceProtection_Yes"] = 1
    elif device_protection == "No internet service":
        row["DeviceProtection_No internet service"] = 1

    # Tech Support
    if tech_support == "Yes":
        row["TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        row["TechSupport_No internet service"] = 1

    # Streaming TV
    if streaming_tv == "Yes":
        row["StreamingTV_Yes"] = 1
    elif streaming_tv == "No internet service":
        row["StreamingTV_No internet service"] = 1

    # Streaming Movies
    if streaming_movies == "Yes":
        row["StreamingMovies_Yes"] = 1
    elif streaming_movies == "No internet service":
        row["StreamingMovies_No internet service"] = 1

    # Contract
    if contract == "One year":
        row["Contract_One year"] = 1
    elif contract == "Two year":
        row["Contract_Two year"] = 1

    # Payment Method
    if payment == "Credit card (automatic)":
        row["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        row["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        row["PaymentMethod_Mailed check"] = 1

    # Tenure Group
    if 13 <= tenure <= 24:
        row["tenure_group_13-24"] = 1
    elif 25 <= tenure <= 48:
        row["tenure_group_25-48"] = 1
    elif 49 <= tenure <= 60:
        row["tenure_group_49-60"] = 1
    elif tenure >= 61:
        row["tenure_group_61-72"] = 1

    X = pd.DataFrame([row])

    try:
        probability = model.predict_proba(X)[0][1]

        if probability >= 0.7:
            risk = "🔴 High Risk"
            action = "Urgent: Offer contract upgrade discount"

        elif probability >= 0.4:
            risk = "🟠 Medium Risk"
            action = "Monitor and re-engage customer"

        else:
            risk = "🟢 Low Risk"
            action = "No action needed"

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        st.write("### Risk Segment")
        st.write(risk)

        st.write("### Recommended Action")
        st.write(action)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
