import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("C:/Users/MukundKumarGupta/Downloads/SG_Files/Employee_Salary_Prediction/best_model.pkl")

# Define prediction function
def predict_income(age, workclass, education, marital_status, occupation,
                   relationship, race, gender, capital_gain, capital_loss,
                   hours_per_week, native_country):

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
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = ">50K" if pred == 1 else "<=50K"
    return f"Prediction: {label}\nConfidence: {proba:.2%}"

# Gradio UI
demo = gr.Interface(
    fn=predict_income,
    inputs=[
        gr.Slider(18, 75, value=30, label="Age"),
        gr.Dropdown(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                     'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'], label="Workclass"),
        gr.Dropdown(['Bachelors', 'Some-college', 'HS-grad', 'Masters', 'Doctorate'], label="Education"),
        gr.Dropdown(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed'], label="Marital Status"),
        gr.Dropdown(['Exec-managerial', 'Prof-specialty', 'Sales', 'Other-service', 'Adm-clerical'], label="Occupation"),
        gr.Dropdown(['Husband', 'Not-in-family', 'Own-child', 'Unmarried'], label="Relationship"),
        gr.Dropdown(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], label="Race"),
        gr.Radio(['Male', 'Female'], label="Gender"),
        gr.Number(label="Capital Gain", value=0),
        gr.Number(label="Capital Loss", value=0),
        gr.Slider(1, 80, value=40, label="Hours per Week"),
        gr.Dropdown(['United-States', 'India', 'Philippines', 'Mexico'], label="Native Country")
    ],
    outputs="text",
    title="💼 Employee Income Prediction",
    description="Fill the details to predict whether an employee earns more than 50K."
)

# Launch the app
demo.launch(share=True)