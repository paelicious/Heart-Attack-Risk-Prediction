import streamlit as st
import numpy as np
import joblib

# 1. Load trained model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Page Configuration
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

st.title("🫀 Heart Attack Risk Prediction System")
st.write("Enter the patient's clinical parameters to predict heart disease risk.")

st.subheader("Patient Clinical Data")

# 3. Patient Input Form (2 Columns Layout)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trtbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

with col2:
    thalachh = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exng = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slp = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    caa = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thall = st.selectbox("Thalassemia Rate", [0, 1, 2, 3])

# 4. Prediction & Output
if st.button("Predict Risk"):
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("⚠️ HIGH RISK: The patient shows a high probability of heart disease risk.")
    else:
        st.success("✅ LOW RISK: The patient shows a low probability of heart disease risk.")