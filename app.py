import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# Load the pre-trained model
model = joblib.load('xgb_model_with_monotonic_constraints.pkl')

# Set the title of the Streamlit app
st.title("Credit Card Fraud Detection System")

# Explanation of the app
st.write("""
    This application predicts whether a credit card transaction is fraudulent based on the transaction's features. 
    Please input the values for each feature below.
""")

# Input fields for transaction features (V1 to V28 + Amount)
input_data = []
for i in range(1, 29):  # Assuming 28 features (V1 to V28)
    feature_value = st.number_input(f'Input feature V{i}', value=0.0)
    input_data.append(feature_value)

# Additional input for 'Amount'
amount = st.number_input('Transaction Amount', value=0.0)
input_data.append(amount)

# Button to predict
if st.button('Predict'):
    input_array = np.array([input_data])  # Convert the input to a NumPy array
    
    # Predict with the model
    prediction = model.predict(input_array)
    
    # Show the prediction result
    if prediction[0] == 0:
        st.success("This is a Normal Transaction.")
    else:
        st.error("This is a Fraudulent Transaction.")
