
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "best_model.joblib"


@st.cache_resource
def load_model():
    """Load the trained prediction pipeline."""
    return joblib.load(MODEL_PATH)


def main():
    """Display the tourism package prediction application."""

    st.set_page_config(
        page_title="Wellness Tourism Predictor",
        page_icon="✈️",
        layout="centered"
    )

    st.title("✈️ Wellness Tourism Package Predictor")

    st.write(
        "Enter customer information to estimate the likelihood "
        "of purchasing the Wellness Tourism Package."
    )

    model = load_model()

    age = st.number_input("Age", 18, 100, 35)
    contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input(
        "Duration of Pitch", 1, 150, 15
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business",
         "Large Business", "Free Lancer"]
    )
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Fe Male"]
    )
    persons = st.number_input(
        "Number of Persons Visiting", 1, 10, 3
    )
    followups = st.number_input(
        "Number of Followups", 1, 10, 4
    )
    product = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard",
         "Super Deluxe", "King"]
    )
    property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )
    marital = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced", "Unmarried"]
    )
    trips = st.number_input(
        "Number of Trips", 1, 30, 3
    )
    passport = st.selectbox("Passport", [0, 1])
    satisfaction = st.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
    )
    own_car = st.selectbox("Own Car", [0, 1])
    children = st.number_input(
        "Number of Children Visiting", 0, 10, 1
    )
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    income = st.number_input(
        "Monthly Income", 1000, 150000, 22000
    )

    if st.button("Predict Purchase", type="primary"):

        input_data = pd.DataFrame([{
            "Age": age,
            "TypeofContact": contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons,
            "NumberOfFollowups": followups,
            "ProductPitched": product,
            "PreferredPropertyStar": property_star,
            "MaritalStatus": marital,
            "NumberOfTrips": trips,
            "Passport": passport,
            "PitchSatisfactionScore": satisfaction,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children,
            "Designation": designation,
            "MonthlyIncome": income
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction")

        if prediction == 1:
            st.success(
                "Customer is predicted to purchase the package."
            )
        else:
            st.info(
                "Customer is predicted not to purchase the package."
            )

        st.metric(
            "Estimated Purchase Probability",
            f"{probability:.2%}"
        )


if __name__ == "__main__":
    main()
