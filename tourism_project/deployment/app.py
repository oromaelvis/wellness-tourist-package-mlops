import os
import joblib
import pandas as pd
import streamlit as st

# must match CLASSIFICATION_THRESHOLD in train.py
CLASSIFICATION_THRESHOLD = 0.40

st.set_page_config(page_title="Wellness Package Predictor", layout="centered")


@st.cache_resource
def load_model():
    # the pipeline commits the trained model next to this file
    model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
    return joblib.load(model_path)


model = load_model()

st.title("Wellness Tourism Package Predictor")
st.write("Internal tool for the Visit with Us sales team. "
         "Enter the customer details below to check how likely they are to buy the Wellness Tourism Package.")

st.subheader("Customer profile")
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=18, max_value=80, value=32)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    MonthlyIncome = st.number_input("Gross monthly income", min_value=1000.0, max_value=100000.0,
                                    value=22000.0, step=500.0)
    CityTier = st.selectbox("City tier", [1, 2, 3], help="1 = most developed, 3 = least developed")
with col2:
    Passport = st.radio("Has a valid passport?", ["Yes", "No"], horizontal=True)
    OwnCar = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)
    NumberOfTrips = st.number_input("Trips per year (average)", min_value=0, max_value=25, value=3)
    PreferredPropertyStar = st.selectbox("Preferred hotel rating", [3, 4, 5])
    NumberOfPersonVisiting = st.number_input("People travelling with the customer", min_value=1, max_value=5, value=3)
    NumberOfChildrenVisiting = st.number_input("Children under 5 travelling", min_value=0, max_value=3, value=1)

st.subheader("Sales interaction")
col3, col4 = st.columns(2)
with col3:
    TypeofContact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    ProductPitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    DurationOfPitch = st.number_input("Pitch duration (minutes)", min_value=1, max_value=130, value=15)
with col4:
    NumberOfFollowups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    PitchSatisfactionScore = st.slider("Pitch satisfaction score", 1, 5, 3)

# put the inputs into a dataframe with the same columns used in training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_data)[0, 1])
    st.metric("Probability of purchase", f"{probability:.0%}")
    if probability >= CLASSIFICATION_THRESHOLD:
        st.success("Likely to BUY. Prioritise this customer for a call and follow-ups.")
    else:
        st.info("Unlikely to buy right now. Keep in the general marketing list.")
    with st.expander("Inputs sent to the model"):
        st.dataframe(input_data.T.rename(columns={0: "value"}).astype(str))
