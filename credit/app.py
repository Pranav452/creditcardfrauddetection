import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = xgb.Booster()
model.load_model('xgboost_model.json')

# Define the feature names (the ones used during training)
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
                 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 
                 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Create a function to make predictions
def predict_fraud(data):
    # Convert the data to the DMatrix format used by XGBoost with feature names
    dmatrix = xgb.DMatrix(data, feature_names=feature_names)
    prediction = model.predict(dmatrix)
    return prediction

# Frontend using Streamlit
st.title('Credit Card Fraud Detection')

# Create input fields for all required features
st.header('Enter Transaction Details:')

# Input fields for Time and Amount
time = st.number_input('Time (seconds since first transaction)', min_value=0, value=0)
amount = st.number_input('Amount', min_value=0.0, value=0.0)

# Input fields for features V1 to V28
v1 = st.number_input('V1', min_value=-100.0, max_value=100.0, value=0.0)
v2 = st.number_input('V2', min_value=-100.0, max_value=100.0, value=0.0)
v3 = st.number_input('V3', min_value=-100.0, max_value=100.0, value=0.0)
v4 = st.number_input('V4', min_value=-100.0, max_value=100.0, value=0.0)
v5 = st.number_input('V5', min_value=-100.0, max_value=100.0, value=0.0)
v6 = st.number_input('V6', min_value=-100.0, max_value=100.0, value=0.0)
v7 = st.number_input('V7', min_value=-100.0, max_value=100.0, value=0.0)
v8 = st.number_input('V8', min_value=-100.0, max_value=100.0, value=0.0)
v9 = st.number_input('V9', min_value=-100.0, max_value=100.0, value=0.0)
v10 = st.number_input('V10', min_value=-100.0, max_value=100.0, value=0.0)
v11 = st.number_input('V11', min_value=-100.0, max_value=100.0, value=0.0)
v12 = st.number_input('V12', min_value=-100.0, max_value=100.0, value=0.0)
v13 = st.number_input('V13', min_value=-100.0, max_value=100.0, value=0.0)
v14 = st.number_input('V14', min_value=-100.0, max_value=100.0, value=0.0)
v15 = st.number_input('V15', min_value=-100.0, max_value=100.0, value=0.0)
v16 = st.number_input('V16', min_value=-100.0, max_value=100.0, value=0.0)
v17 = st.number_input('V17', min_value=-100.0, max_value=100.0, value=0.0)
v18 = st.number_input('V18', min_value=-100.0, max_value=100.0, value=0.0)
v19 = st.number_input('V19', min_value=-100.0, max_value=100.0, value=0.0)
v20 = st.number_input('V20', min_value=-100.0, max_value=100.0, value=0.0)
v21 = st.number_input('V21', min_value=-100.0, max_value=100.0, value=0.0)
v22 = st.number_input('V22', min_value=-100.0, max_value=100.0, value=0.0)
v23 = st.number_input('V23', min_value=-100.0, max_value=100.0, value=0.0)
v24 = st.number_input('V24', min_value=-100.0, max_value=100.0, value=0.0)
v25 = st.number_input('V25', min_value=-100.0, max_value=100.0, value=0.0)
v26 = st.number_input('V26', min_value=-100.0, max_value=100.0, value=0.0)
v27 = st.number_input('V27', min_value=-100.0, max_value=100.0, value=0.0)
v28 = st.number_input('V28', min_value=-100.0, max_value=100.0, value=0.0)

# Create a button to make a prediction
if st.button('Predict Fraud'):
    # Prepare data for prediction
    data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]])
    
    # Convert data to DataFrame with appropriate column names
    data_df = pd.DataFrame(data, columns=feature_names)
    
    # In a real case, you'll need to normalize/standardize input using a scaler (like you did during training)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_df)

    # Convert to DataFrame again (as scaling may remove the column names)
    data_scaled_df = pd.DataFrame(data_scaled, columns=feature_names)

    # Make the prediction
    prediction = predict_fraud(data_scaled_df)

    # Show the result
    if prediction > 0.5:
        st.write('⚠️ Fraudulent Transaction Detected!')
    else:
        st.write('✅ Transaction Seems Legitimate')
