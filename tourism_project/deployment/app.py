
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------ LOAD THE TRAINED MODEL FROM THE HF MODEL HUB ------------------
model_path = hf_hub_download(
    repo_id="ivkobenko/tourism-package-model",
    filename="best_tourism_package_model_v1.joblib",
)
model = joblib.load(model_path)

# ---------------- APP HEADER ------------------
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the newly
introduced **Wellness Tourism Package**, helping 'Visit with Us' target the
right customers before contacting them. Enter the customer details below.
""")
# Just text on the page

# -------------- CUSTOMER DETAILS -------------
st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
#  A number field. min_value/max_value - user physically cannot enter age 500. 
# value=35 = default shown at start. The user's answer is saved into the variable age.

typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
# dropdown list. The user can only pick from the given options — no typos possible.

num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.number_input("Number of Trips (per year)", min_value=0, max_value=30, value=2)
passport = st.selectbox("Has Passport?", ["Yes", "No"])
own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
num_children = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=20000.0, step=100.0)

# ------------- CUSTOMER INTERACTION DATA ------------------
st.header("Customer Interaction Data")
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
pitch_duration = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

# ------------ ASSEMBLE INPUTS INTO A DATAFRAME ------------------
# Column names must exactly match the training feature names
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": pitch_duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])
# Pack the answers into a table



# ------------------ PREDICTION ------------------

# Same classification threshold as used during model evaluation
CLASSIFICATION_THRESHOLD = 0.45

# The button and the prediction
if st.button("Predict Purchase"):
    proba = model.predict_proba(input_data)[0, 1]
    prediction = int(proba >= CLASSIFICATION_THRESHOLD)
# if the probability is 45% or more -  count as "will buy"

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The customer is **LIKELY to purchase** the Wellness Tourism Package (probability: {proba:.2%})")
    else:
        st.info(f"The customer is **UNLIKELY to purchase** the Wellness Tourism Package (probability: {proba:.2%})")
