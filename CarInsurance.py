# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load the saved model
# import plotly as plt
# import seaborn as sns
# import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
# Load the trained model
model = joblib.load('car_insurance_model.pkl')

# Title of the web app
st.title("Car Insurance Prediction")

# User inputs
INCOME = st.number_input("Enter Income:",value=0)
HOME_VAL = st.number_input("Enter your Home value:",value=0)
CLAIM_FLAG = st.selectbox("Claim Flag (Encoded):", [0, 1])
CAR_AGE = st.number_input("Enter Car Age:",value=0)
YOJ = st.number_input("Enter YOJ value:",value=0)
MVR_PTS = st.number_input("Enter MVR_PTS value:",value=0)
CLM_FREQ = st.number_input("Enter your claim frequency:",value=0)
TIF = st.number_input("Enter your TIF value:",value=0)
BLUEBOOK = st.number_input("Enter Bluebook value:",value=0)
OLDCLAIM = st.number_input("Enter Old Claim Amount:",value=0)
TRAVTIME = st.number_input("Enter Travel time of car:",value=0)

# Prepare the input for the model
input_data = pd.DataFrame({'CLAIM_FLAG': [CLAIM_FLAG],'BLUEBOOK': [BLUEBOOK],'INCOME': [INCOME],'OLDCLAIM': [OLDCLAIM],'TRAVTIME': [TRAVTIME],'HOME_VAL': [HOME_VAL],'CAR_AGE': [CAR_AGE],'YOJ': [YOJ],'MVR_PTS': [MVR_PTS],'CLM_FREQ': [CLM_FREQ],'TIF': [TIF]})

# Predict the insurance outcome using the loaded model
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The predicted car insurance outcome is: {prediction[0]}")
