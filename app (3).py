
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title='Salary Predictor', page_icon='💼', layout='centered')

# Load the model and encoder
@st.cache_resource
def load_assets():
    model = pickle.load(open('best_salary_model.pkl', 'rb'))
    encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, encoder

try:
    model, encoder = load_assets()
except Exception as e:
    st.error("Error loading model files. Please ensure .pkl files are in the same folder.")

st.title('📈 Professional Salary Estimator')
st.markdown("--- ")
st.write("Predict your potential salary based on market data using our Random Forest model.")

# Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=18, max_value=80, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        education = st.selectbox('Education Level', ["Bachelor's", "Master's", "PhD", "High School"])

    with col2:
        experience = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, value=5.0)
        job_title = st.text_input('Job Title', 'Software Engineer')

st.markdown("--- ")

if st.button('Calculate Estimated Salary', use_container_width=True):
    # Prepare input
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': experience
    }])

    # Encoding
    for col in ['Gender', 'Education Level', 'Job Title']:
        try:
            input_df[col] = encoder.transform(input_df[col])
        except:
            input_df[col] = 0 

    # Prediction
    prediction = model.predict(input_df)[0]
    
    st.metric(label="Estimated Annual Salary", value=f"${prediction:,.2f}")
    st.success("Prediction complete!")
    st.balloons()
