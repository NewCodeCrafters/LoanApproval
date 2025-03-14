import streamlit as st
import pandas as pd
import joblib

st.title('Loan Approval Prediction App')

@st.cache_data
def load_data():
    return pd.read_csv('data/loan_features.csv')

@st.cache_resource
def load_model():
    with open('model/model.joblib', 'rb') as f:
        model = joblib.load(f)
    return model

@st.cache_resource
def load_scaler_encoder():
    with open('model/scaler_encode.joblib', 'rb') as f:
        scaler_encode =  joblib.load(f)
    return scaler_encode

data = load_data()
model = load_model()
scaler_encoder = load_scaler_encoder()

person_gender = st.selectbox("Gender", options=data['person_gender'].unique())
person_education = st.selectbox("Education", options=data['person_education'].unique())
person_home_ownership = st.selectbox("Home Ownership", options=data['person_home_ownership'].unique())
loan_intent = st.selectbox("Loan Intent", options=data['loan_intent'].unique())
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", options=data['previous_loan_defaults_on_file'].unique())
person_age = st.number_input("Age", min_value=18, max_value=120, value=18) 
person_income = st.number_input("Income", min_value=0.0, max_value=data['person_income'].max(), value=0.0)
person_emp_exp = st.slider("Employment Experience", min_value=0, max_value=30, value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0.0, max_value=data['loan_amnt'].max(), value=0.0)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=data['loan_int_rate'].max(), value=0.0)
loan_percent_income = st.number_input("Loan Per Percent Income", min_value=0.0, max_value=data['loan_percent_income'].max(), value=0.0)
cb_person_cred_hist_length 	= st.slider("Credit History Length", min_value=0.0, max_value=data['cb_person_cred_hist_length'].max(), value=0.0)
credit_score = st.number_input("Credit Score", min_value=0, max_value=data['credit_score'].max(), value=0)

button = st.button("Predict Loan Approval")

if button:
    features = pd.DataFrame({
        "person_gender": person_gender,
        "person_education": person_education,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score
    }, index=[0])
    
    features = scaler_encoder.transform(features)
    prediction = model.predict(features)
    output_dictionary = {0: "Rejected", 1: "Approved"}
    st.write(f"Loan Approval Prediction: {output_dictionary[prediction[0]]}")
