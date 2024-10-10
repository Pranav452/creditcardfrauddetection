import streamlit as st
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

try:
    import xgboost as xgb
    st.success("XGBoost imported successfully!")
except ImportError as e:
    st.error(f"Failed to import XGBoost: {e}")
    st.info("Please make sure XGBoost is installed correctly.")

try:
    from sklearn.preprocessing import StandardScaler
    st.success("StandardScaler imported successfully!")
except ImportError as e:
    st.error(f"Failed to import StandardScaler: {e}")
    st.info("Please make sure scikit-learn is installed correctly.")

# Display Python and package versions
import sys
st.write(f"Python version: {sys.version}")
st.write(f"NumPy version: {np.__version__}")
st.write(f"Pandas version: {pd.__version__}")

# Load the trained model
model = None
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'xgboost_model.json')

print("Current working directory:", os.getcwd())
print("Attempting to load model from:", os.path.abspath(model_path))

if os.path.exists(model_path):
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.info("Please make sure the model file is not corrupted.")
else:
    st.error(f"Model file '{model_path}' not found.")
    st.info("Please make sure the model file is in the correct location.")

# Define the feature names (the ones used during training)
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
                 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 
                 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Create a function to make predictions
def predict_fraud(data):
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
        return None
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
v_inputs = []
for i in range(1, 29):
    v_inputs.append(st.number_input(f'V{i}', min_value=-100.0, max_value=100.0, value=0.0))

# Create a button to make a prediction
if st.button('Predict Fraud'):
    # Prepare data for prediction
    data = np.array([[time] + v_inputs + [amount]])
    
    # Convert data to DataFrame with appropriate column names
    data_df = pd.DataFrame(data, columns=feature_names)
    
    # Normalize/standardize input
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_df)

    # Convert to DataFrame again (as scaling may remove the column names)
    data_scaled_df = pd.DataFrame(data_scaled, columns=feature_names)

    # Make the prediction
    prediction = predict_fraud(data_scaled_df)

    # Show the result
    if prediction is not None:
        if prediction[0] > 0.5:
            st.warning('⚠️ Potential Fraudulent Transaction Detected!')
        else:
            st.success('✅ Transaction Seems Legitimate')
        st.write(f"Fraud Probability: {prediction[0]:.2%}")
    else:
        st.error("Unable to make prediction due to model loading issues.")

st.info("Note: This is a demo application. In a real-world scenario, the model would be properly trained and validated, and the input data would be preprocessed consistently with the training data.")