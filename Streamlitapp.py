import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model
import joblib

model = joblib.load("C:/Users/MukundKumarGupta/Downloads/SG_Files/Employee_Salary_Prediction/best_model.pkl")

st.title("Employee Income Prediction App")

st.write("Fill the form below to predict whether an employee earns >50K")

# Input fields
age = st.slider("Age", 18, 75, 30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                                       'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate',
                                       '5th-6th', 'Preschool'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                         'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                         'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.radio("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0)
capital_loss = st.number_input("Capital Loss", min_value=0)
hours_per_week = st.slider("Hours per Week", 1, 80, 40)
native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada',
                                                 'Puerto-Rico', 'El-Salvador', 'China', 'Cuba', 'England'])

# Create dataframe
data = {
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [np.log1p(capital_gain)],
    'capital-loss': [np.log1p(capital_loss)],
    'hours-per-week': [min(hours_per_week, 80)],
    'native-country': [native_country]
}

input_df = pd.DataFrame(data)

# Predict
if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This employee is predicted to earn **more than $50K**.")
    else:
        st.info("ℹ️ This employee is predicted to earn **less than or equal to $50K**.")