
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and encoder
try:
    model = pickle.load(open('best_salary_model.pkl', 'rb'))
    encoder = pickle.load(open('label_encoder.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or Encoder files not found. Please ensure 'best_salary_model.pkl' and 'label_encoder.pkl' are in the same directory.")

st.title('💵 Salary Prediction App')
st.write('Enter professional details to estimate the expected salary based on our Random Forest model.')

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    education = st.selectbox('Education Level', ["Bachelor's", "Master's", "PhD", "High School"])

with col2:
    experience = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    job_title = st.text_input('Job Title', 'Software Engineer')

if st.button('Predict Salary'):
    # Prepare input data
    input_dict = {
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': experience
    }
    
    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    # Note: Using a simple transform; in production, you'd handle unknown categories more robustly
    for col in ['Gender', 'Education Level', 'Job Title']:
        try:
            input_df[col] = encoder.transform(input_df[col])
        except:
            # Fallback to a default value if category is unknown
            input_df[col] = 0 

    # Make prediction
    prediction = model.predict(input_df)
    
    st.success(f'### Estimated Salary: ${prediction[0]:,.2f}')
    st.balloons()
