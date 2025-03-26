import streamlit as st
import joblib
import numpy as np

# Load model dan encoder
model = joblib.load('finalmodel.pkl')
gender_enc = joblib.load('gender_encode.pkl')
contract_enc = joblib.load('Contract_Length_encode.pkl')
subscription_enc = joblib.load('Subscription_encode.pkl')

def preprocess_input(user_input):
    """ Encode categorical values using pre-trained encoders. """
    user_input[0] = gender_enc["Gender"].get(user_input[0], 0)  # Encode Gender
    user_input[6] = subscription_enc["Subscription"].get(user_input[6], 0)  # Encode Subscription
    user_input[7] = contract_enc["Contract_Length"].get(user_input[7], 0)  # Encode Contract Length
    
    return np.array(user_input, dtype=float).reshape(1, -1)  # Convert to float

def make_prediction(features):
    """ Use the trained model to make predictions. """
    input_array = preprocess_input(features)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Masukkan data pelanggan untuk memprediksi apakah pelanggan akan churn atau tidak.")

# User input dengan slider (disesuaikan dengan dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=65, value=30)  # 18 - 65
tenure = st.slider("Tenure (months)", min_value=1, max_value=60, value=24)  # 1 - 60
usage_frequency = st.slider("Usage Frequency", min_value=1, max_value=30, value=10)  # 1 - 30
support_calls = st.slider("Support Calls", min_value=0, max_value=10, value=5)  # 0 - 10
payment_delay = st.slider("Payment Delay", min_value=0, max_value=30, value=3)  # 0 - 30
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.slider("Total Spend", min_value=100, max_value=1000, value=500)  # 100 - 1000
last_interaction = st.slider("Last Interaction (days ago)", min_value=1, max_value=30, value=10, step=1)  # 1 - 30

if st.button("Predict Churn"):
    user_input = [gender, age, tenure, usage_frequency, support_calls, payment_delay,
                  subscription_type, contract_length, total_spend, last_interaction]
    
    prediction = make_prediction(user_input)
    
    if prediction == 1:
        st.error("⚠️ Pelanggan Berisiko Churn!")
    else:
        st.success("✅ Pelanggan Diperkirakan Tetap Bertahan.")
