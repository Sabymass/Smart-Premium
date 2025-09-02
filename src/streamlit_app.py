# src/app/streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title='SmartPremium', layout='centered')
st.title('SmartPremium — Insurance Premium Predictor')

model_file = Path('outputs/smartpremium_pipeline.joblib')
if not model_file.exists():
    st.warning("Model not found. Run training script first:\n`python -m src.models.train_and_evaluate`")
    st.stop()

pipe = joblib.load(model_file)
preproc = pipe['preprocessor']
model = pipe['model']

st.sidebar.header('Customer Inputs')

# Optional numeric inputs: allow empty by setting value=None
Age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=None)
Annual_Income = st.sidebar.number_input('Annual Income', min_value=0, max_value=10**7, value=None)
Number_of_Dependents = st.sidebar.number_input('Number of Dependents', min_value=0, max_value=10, value=None)
Health_Score = st.sidebar.number_input('Health Score', min_value=0.0, max_value=100.0, value=None, step=0.1, format="%.1f")
Previous_Claims = st.sidebar.number_input('Previous Claims', min_value=0, max_value=50, value=None)
Vehicle_Age = st.sidebar.number_input('Vehicle Age', min_value=0, max_value=100, value=None)
Credit_Score = st.sidebar.number_input('Credit Score', min_value=300, max_value=900, value=None)
Insurance_Duration = st.sidebar.number_input('Insurance Duration (yrs)', min_value=0, max_value=100, value=None)

# Select boxes remain the same
Gender = st.sidebar.selectbox('Gender', ['Male','Female'])
Marital_Status = st.sidebar.selectbox('Marital Status', ['Single','Married','Divorced'])
Education_Level = st.sidebar.selectbox('Education Level', ['High School', "Bachelor's", "Master's", 'PhD'])
Occupation = st.sidebar.selectbox('Occupation', ['Employed','Self-Employed','Unemployed'])
Location = st.sidebar.selectbox('Location', ['Urban','Suburban','Rural'])
Policy_Type = st.sidebar.selectbox('Policy Type', ['Basic','Comprehensive','Premium'])
Smoking_Status = st.sidebar.selectbox('Smoking Status', ['Yes','No'])
Exercise_Frequency = st.sidebar.selectbox('Exercise Frequency', ['Daily','Weekly','Monthly','Rarely'])
Property_Type = st.sidebar.selectbox('Property Type', ['House','Apartment','Condo'])

if st.button('Predict'):
    # Use fallback defaults if inputs are empty
    df = pd.DataFrame([{
        'Age': Age if Age is not None else 30,
        'Gender': Gender,
        'Annual Income': Annual_Income if Annual_Income is not None else 30000,
        'Marital Status': Marital_Status,
        'Number of Dependents': Number_of_Dependents if Number_of_Dependents is not None else 0,
        'Education Level': Education_Level,
        'Occupation': Occupation,
        'Health Score': Health_Score if Health_Score is not None else 70.0,
        'Location': Location,
        'Policy Type': Policy_Type,
        'Previous Claims': Previous_Claims if Previous_Claims is not None else 0,
        'Vehicle Age': Vehicle_Age if Vehicle_Age is not None else 3,
        'Credit Score': Credit_Score if Credit_Score is not None else 650,
        'Insurance Duration': Insurance_Duration if Insurance_Duration is not None else 1,
        'Policy Start Date': None,
        'Customer Feedback': 'N/A',
        'Smoking Status': Smoking_Status,
        'Exercise Frequency': Exercise_Frequency,
        'Property Type': Property_Type
    }])
    X_proc = preproc.transform(df)
    pred = model.predict(X_proc)[0]
    st.success(f'Predicted Premium Amount: {pred:,.2f}')
