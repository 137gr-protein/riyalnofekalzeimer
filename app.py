# Coba without Country
import streamlit as st
import pandas as pd
import joblib

# Load model and pipeline
model = joblib.load("model_alzheimer_no_country.pkl")
pipeline = joblib.load("preprocessing_for_ML.pkl")

st.title("🧠 Alzheimer's Disease Prediction")
st.write("Enter the details below to predict Alzheimer's risk:")

# Input fields with min and max values from your data
age = st.number_input("Age", min_value=50, max_value=94, value=65)
gender = st.number_input("Gender (0: Female, 1: Male)", min_value=0, max_value=1, value=0)
education_level = st.number_input("Education Level (0–19)", min_value=0, max_value=19, value=12)
bmi = st.number_input("BMI", min_value=18.5, max_value=35.0, value=22.0)
physical_activity = st.number_input("Physical Activity Level (0: Low, 1: Moderate, 2: High)", min_value=0, max_value=2, value=1)
smoking_status = st.number_input("Smoking Status (0: Never, 1: Former, 2: Current)", min_value=0, max_value=2, value=0)
alcohol = st.number_input("Alcohol Consumption (0: None, 1: Low, 2: Moderate)", min_value=0, max_value=2, value=1)
diabetes = st.number_input("Diabetes (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
hypertension = st.number_input("Hypertension (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
cholesterol = st.number_input("Cholesterol Level (0: Normal, 1: High)", min_value=0, max_value=1, value=0)
family_history = st.number_input("Family History of Alzheimer’s (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
cognitive_score = st.number_input("Cognitive Test Score", min_value=30, max_value=99, value=70)
depression_level = st.number_input("Depression Level (0: None, 1: Mild, 2: Moderate)", min_value=0, max_value=2, value=1)
sleep_quality = st.number_input("Sleep Quality (0: Low, 1: Medium, 2: High)", min_value=0, max_value=2, value=1)
dietary_habits = st.number_input("Dietary Habits (0: Unhealthy, 1: Moderate, 2: Healthy)", min_value=0, max_value=2, value=1)
air_pollution = st.number_input("Air Pollution Exposure (0: Low, 1: Medium, 2: High)", min_value=0, max_value=2, value=1)
genetic_risk = st.number_input("Genetic Risk Factor (APOE-ε4 allele) (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
social_engagement = st.number_input("Social Engagement Level (0: Low, 1: Medium, 2: High)", min_value=0, max_value=2, value=1)
income_level = st.number_input("Income Level (0: Low, 1: Middle, 2: High)", min_value=0, max_value=2, value=1)
stress_levels = st.number_input("Stress Levels (0: Low, 1: Medium, 2: High)", min_value=0, max_value=2, value=1)
urban_rural = st.number_input("Urban vs Rural Living (0: Urban, 1: Rural)", min_value=0, max_value=1, value=0)
employment_employed = st.number_input("Employment Status: Employed (0 or 1)", min_value=0, max_value=1, value=1)
employment_unemployed = st.number_input("Employment Status: Unemployed (0 or 1)", min_value=0, max_value=1, value=0)
employment_retired = st.number_input("Employment Status: Retired (0 or 1)", min_value=0, max_value=1, value=0)
marital_married = st.number_input("Marital Status: Married (0 or 1)", min_value=0, max_value=1, value=0)
marital_single = st.number_input("Marital Status: Single (0 or 1)", min_value=0, max_value=1, value=1)
marital_widowed = st.number_input("Marital Status: Widowed (0 or 1)", min_value=0, max_value=1, value=0)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'BMI': bmi,
        'Physical Activity Level': physical_activity,
        'Smoking Status': smoking_status,
        'Alcohol Consumption': alcohol,
        'Diabetes': diabetes,
        'Hypertension': hypertension,
        'Cholesterol Level': cholesterol,
        'Family History of Alzheimer’s': family_history,
        'Cognitive Test Score': cognitive_score,
        'Depression Level': depression_level,
        'Sleep Quality': sleep_quality,
        'Dietary Habits': dietary_habits,
        'Air Pollution Exposure': air_pollution,
        'Genetic Risk Factor (APOE-ε4 allele)': genetic_risk,
        'Social Engagement Level': social_engagement,
        'Income Level': income_level,
        'Stress Levels': stress_levels,
        'Urban vs Rural Living': urban_rural,
        'Employment Status_Employed': employment_employed,
        'Employment Status_Unemployed': employment_unemployed,
        'Employment Status_Retired': employment_retired,
        'Marital Status_Married': marital_married,
        'Marital Status_Single': marital_single,
        'Marital Status_Widowed': marital_widowed,
    }])

    # transformed = pipeline.transform(input_df)
    # prediction = model.predict(transformed)[0]
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Alzheimer's")
    else:
        st.success("✅ Low Risk of Alzheimer's")
