import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the trained model
model = joblib.load("upi_fraud_model.pkl")

# Initialize the label encoder
label_encoder = LabelEncoder()

# Streamlit UI components
st.title("UPI Fraud Detection App")

# Input fields for transaction details
transaction_type = st.selectbox("Enter transaction type", ["Fund Transfer", "Bill Payment", "Recharge", "Other"])
amount = st.number_input("Enter transaction amount", min_value=1)
location = st.text_input("Enter transaction location")
time = st.text_input("Enter transaction time (HH:MM format)")

# Encode the transaction type to a numerical value
transaction_type_encoded = label_encoder.fit_transform([transaction_type])[0]

# Encode the location to a numerical value (this assumes that the model expects a numerical value for location)
location_encoded = label_encoder.fit_transform([location])[0]

# Convert time (HH:MM) to a numerical value (e.g., hour of the day)
try:
    time_obj = datetime.strptime(time, "%H:%M")  # Convert string to datetime object
    hour = time_obj.hour  # Extract the hour part as the numeric feature
except ValueError:
    st.write("Please enter the time in HH:MM format.")
    hour = 0  # Default value in case of error

# You may need to add more features here, depending on the training data
# Example: Adding dummy values for other missing features
additional_features = [0] * (30 - 4)  # Placeholder for other 26 features not included

# Combine all features into one list
features = [transaction_type_encoded, amount, location_encoded, hour] + additional_features

# Predict fraud when the button is clicked
if st.button("Predict Fraud"):
    # Make the fraud prediction
    prediction = model.predict([features])
    
    # Show the prediction result
    if prediction == 1:
        st.write("Fraud Detected!")
    else:
        st.write("Transaction is Safe")
