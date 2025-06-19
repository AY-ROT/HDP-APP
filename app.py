# Set page config immediately
import streamlit as st
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Core imports
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# -------------------- Load Logistic Regression Model --------------------
@st.cache_resource
def load_model():
    with open("LogisticR.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------- Download Utility --------------------
def download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

# -------------------- Input Encoder --------------------
def encode_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                 resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):

    sex = 0 if sex == "Male" else 1
    cp = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fbs = 1 if fasting_bs == "> 120 mg/dl" else 0
    recg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exang = 1 if exercise_angina == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    return pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [cp],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fbs],
        'RestingECG': [recg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exang],
        'Oldpeak': [oldpeak],
        'ST_Slope': [slope]
    })

# -------------------- Header --------------------
st.title("💓 Heart Disease Prediction System")
st.markdown("Welcome to the **Heart Disease Predictor**, powered by a trained Logistic Regression model. Use the tabs below to test individuals, upload multiple cases, or view model details.")

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Bulk Prediction", "📊 Model Info"])

# ----------- TAB 1: Single Patient Prediction (Updated with Descriptions) -----------
with tab1:
    st.subheader("Patient Information")
    st.markdown("Please fill in the following details about the patient. Each field has a description to guide you.")

    age = st.number_input("Age", 0, 120, help="Enter the patient's age in years.")
    sex = st.radio("Sex", ["Male", "Female"], help="Select the biological sex of the patient.")
    chest_pain = st.selectbox(
        "Chest Pain Type", 
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        help="""Choose the type of chest pain:
        - Typical Angina: Predictable chest pain during exertion  
        - Atypical Angina: Unpredictable chest pain  
        - Non-Anginal Pain: Chest pain not related to the heart  
        - Asymptomatic: No chest pain at all"""
    )
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 0, 300, help="The patient’s blood pressure when resting, in mmHg.")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 0, help="The level of cholesterol in the blood (mg/dL).")
    fasting_bs = st.radio(
        "Fasting Blood Sugar", 
        ["<= 120 mg/dl", "> 120 mg/dl"],
        help="Was the patient's fasting blood sugar above 120 mg/dl? Choose '> 120 mg/dl' if yes."
    )
    resting_ecg = st.selectbox(
        "Resting ECG", 
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        help="""Result of resting electrocardiogram:
        - Normal: No abnormality  
        - ST-T Wave Abnormality: Minor changes in ECG  
        - Left Ventricular Hypertrophy: Thickened heart wall"""
    )
    max_hr = st.number_input("Maximum Heart Rate", 60, 220, help="The highest heart rate recorded during exercise.")
    exercise_angina = st.radio(
        "Exercise-Induced Angina", 
        ["No", "Yes"],
        help="Did the patient experience chest pain due to exercise?"
    )
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, help="ST depression induced by exercise, indicating possible blockage.")
    st_slope = st.selectbox(
        "ST Slope", 
        ["Upsloping", "Flat", "Downsloping"],
        help="""The slope of the ST segment during exercise:
        - Upsloping: Normal  
        - Flat: Could indicate a problem  
        - Downsloping: Often associated with heart issues"""
    )

    if st.button("🧠 Predict"):
        data = encode_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
        prediction = model.predict(data)[0]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("❌ High Risk: Heart Disease Likely")
        else:
            st.success("✅ Low Risk: No Heart Disease Detected")

# ----------- TAB 2: Bulk Prediction from CSV -----------
with tab2:
    st.subheader("Batch Prediction from Uploaded CSV")

    st.markdown("""
    Upload a CSV file with the following **11 columns**, encoded numerically:
    ```
    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    ```
    - Make sure all values are numeric and follow encoding:
        - Sex: 0 = Male, 1 = Female  
        - ChestPainType: 0 = Atypical, 1 = Non-Anginal, 2 = Asymptomatic, 3 = Typical  
        - FastingBS: 0 or 1  
        - RestingECG: 0 = Normal, 1 = ST-T Abnormality, 2 = LVH  
        - ExerciseAngina: 0 = No, 1 = Yes  
        - ST_Slope: 0 = Upsloping, 1 = Flat, 2 = Downsloping
    """)

    file = st.file_uploader("📁 Upload your CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        required = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        if all(col in df.columns for col in required):
            try:
                df['Prediction'] = model.predict(df[required])
                df['Result'] = df['Prediction'].map({0: '✅ No Disease', 1: '❌ Disease Detected'})
                st.success("✅ Predictions successful!")
                st.write(df)
                st.markdown(download_link(df), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("⚠️ Incorrect or missing columns.")

# ----------- TAB 3: Model Info -----------
with tab3:
    st.subheader("About the Model")

    st.markdown("""
    This prediction system is powered by a **Logistic Regression** model trained on the **UCI Cleveland Heart Disease dataset**.  
    The model was selected based on:
    - High accuracy (85.86%)
    - Strong ROC-AUC score (0.913)
    - Fast performance and interpretability

    **Features used for training** include:
    - Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    """)

    scores = {
        "Accuracy": 85.86,
        "F1-Score": 86.00,
        "ROC-AUC": 91.30
    }
    df_score = pd.DataFrame.from_dict(scores, orient='index', columns=["Score"]).reset_index().rename(columns={"index": "Metric"})

    fig = px.bar(df_score, x="Metric", y="Score", text="Score", color="Metric", title="📈 Logistic Regression Performance")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Logistic Regression was selected for deployment due to its balance of speed, accuracy, and transparency.")

