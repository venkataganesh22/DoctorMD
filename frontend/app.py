import streamlit as st
import requests

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Multi Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🩺 Multi Disease Prediction")
st.write("Select a disease and enter the patient's information.")


# --------------------------------------------------
# SELECT DISEASE
# --------------------------------------------------

disease = st.selectbox(
    "Select Disease",
    ["Heart Disease", "Diabetes"]
)


# ==================================================
# HEART DISEASE
# ==================================================

if disease == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    st.subheader("Patient Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=None,
        placeholder="Enter age",
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["Select", "Female", "Male"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["Select", 0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=200,
        value=None,
        placeholder="Enter blood pressure",
        step=1
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=None,
        placeholder="Enter cholesterol",
        step=1
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["Select", "No", "Yes"]
    )

    restecg = st.selectbox(
        "Resting ECG",
        ["Select", 0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=None,
        placeholder="Enter maximum heart rate",
        step=1
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["Select", "No", "Yes"]
    )

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=None,
        placeholder="Enter ST depression",
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        ["Select", 0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        ["Select", 0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        ["Select", 0, 1, 2, 3]
    )

    # --------------------------------------------------
    # HEART PREDICTION
    # --------------------------------------------------

    if st.button("🔍 Predict Heart Disease", use_container_width=True):

        # Validate every field

        if age is None:
            st.warning("Please enter Age.")
            st.stop()

        if sex == "Select":
            st.warning("Please select Sex.")
            st.stop()

        if cp == "Select":
            st.warning("Please select Chest Pain Type.")
            st.stop()

        if trestbps is None:
            st.warning("Please enter Resting Blood Pressure.")
            st.stop()

        if chol is None:
            st.warning("Please enter Cholesterol.")
            st.stop()

        if fbs == "Select":
            st.warning("Please select Fasting Blood Sugar.")
            st.stop()

        if restecg == "Select":
            st.warning("Please select Resting ECG.")
            st.stop()

        if thalach is None:
            st.warning("Please enter Maximum Heart Rate.")
            st.stop()

        if exang == "Select":
            st.warning("Please select Exercise Induced Angina.")
            st.stop()

        if oldpeak is None:
            st.warning("Please enter ST Depression.")
            st.stop()

        if slope == "Select":
            st.warning("Please select Slope.")
            st.stop()

        if ca == "Select":
            st.warning("Please select Number of Major Vessels.")
            st.stop()

        if thal == "Select":
            st.warning("Please select Thalassemia.")
            st.stop()

        # Convert Yes/No values to 0/1

        sex_value = 1 if sex == "Male" else 0
        fbs_value = 1 if fbs == "Yes" else 0
        exang_value = 1 if exang == "Yes" else 0

        # Data sent to FastAPI

        data = {
            "age": age,
            "sex": sex_value,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_value,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang_value,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        # Call backend

        try:

            response = requests.post(
                f"{API_URL}/predict/heart",
                json=data,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                # Check backend error

                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                    st.stop()

                prediction = result["prediction"]

                st.divider()
                st.subheader("❤️ Prediction Result")

                if prediction == 1:

                    st.error(
                        "⚠️ High Risk of Heart Disease"
                    )

                    st.write(
                        "The model predicts a possible risk "
                        "of heart disease."
                    )

                else:

                    st.success(
                        "✅ Low Risk of Heart Disease"
                    )

                    st.write(
                        "The model predicts a lower risk "
                        "of heart disease."
                    )

            else:

                st.error(
                    f"Backend returned status code "
                    f"{response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to the backend. "
                "Please start FastAPI first."
            )

        except requests.exceptions.Timeout:

            st.error(
                "❌ The backend took too long to respond."
            )


# ==================================================
# DIABETES
# ==================================================

else:

    st.header("🩸 Diabetes Prediction")

    st.subheader("Patient Information")

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=None,
        placeholder="Enter number of pregnancies",
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=None,
        placeholder="Enter glucose level",
        step=0.1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=None,
        placeholder="Enter blood pressure",
        step=0.1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=None,
        placeholder="Enter skin thickness",
        step=0.1
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=None,
        placeholder="Enter insulin level",
        step=0.1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=None,
        placeholder="Enter BMI",
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=None,
        placeholder="Enter diabetes pedigree function",
        step=0.01
    )

    diabetes_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=None,
        placeholder="Enter age",
        step=1
    )

    # --------------------------------------------------
    # DIABETES PREDICTION
    # --------------------------------------------------

    if st.button("🔍 Predict Diabetes", use_container_width=True):

        # Validate every field

        if pregnancies is None:
            st.warning("Please enter Pregnancies.")
            st.stop()

        if glucose is None:
            st.warning("Please enter Glucose.")
            st.stop()

        if blood_pressure is None:
            st.warning("Please enter Blood Pressure.")
            st.stop()

        if skin_thickness is None:
            st.warning("Please enter Skin Thickness.")
            st.stop()

        if insulin is None:
            st.warning("Please enter Insulin.")
            st.stop()

        if bmi is None:
            st.warning("Please enter BMI.")
            st.stop()

        if diabetes_pedigree is None:
            st.warning(
                "Please enter Diabetes Pedigree Function."
            )
            st.stop()

        if diabetes_age is None:
            st.warning("Please enter Age.")
            st.stop()

        # Data sent to FastAPI

        data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": diabetes_age
        }

        # Call backend

        try:

            response = requests.post(
                f"{API_URL}/predict/diabetes",
                json=data,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                    st.stop()

                prediction = result["prediction"]

                st.divider()
                st.subheader("🩸 Prediction Result")

                if prediction == 1:

                    st.error(
                        "⚠️ High Risk of Diabetes"
                    )

                    st.write(
                        "The model predicts a possible risk "
                        "of diabetes."
                    )

                else:

                    st.success(
                        "✅ Low Risk of Diabetes"
                    )

                    st.write(
                        "The model predicts a lower risk "
                        "of diabetes."
                    )

            else:

                st.error(
                    f"Backend returned status code "
                    f"{response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to the backend. "
                "Please start FastAPI first."
            )

        except requests.exceptions.Timeout:

            st.error(
                "❌ The backend took too long to respond."
            )