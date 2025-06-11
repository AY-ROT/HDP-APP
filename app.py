# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# ===================== Utilities =====================
# Cache the model loading to avoid reloading every time
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to generate a download link for DataFrame as CSV
def download_link(df, filename='predictions.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Predictions CSV</a>'

# ===================== Data Preprocessing =====================
# Converts user-friendly categorical inputs into machine-readable format
def encode_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                 resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):

    sex = 0 if sex == "Male" else 1
    chest_map = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"]
    cp = chest_map.index(chest_pain)
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

# ===================== Prediction =====================
# Predicts with all selected models and returns a list of predictions
def predict_with_all_models(input_df, model_files):
    results = []
    for model_file in model_files:
        try:
            model = load_model(model_file)
            prediction = model.predict(input_df)[0]  # Predict the first (and only) sample
            results.append(prediction)
        except Exception as e:
            st.error(f"Error with model {model_file}: {e}")
            results.append(None)
    return results

# ===================== Streamlit Interface =====================
# Set the page title and layout
st.set_page_config(page_title="Heart Disease Predictor")
st.markdown("""
    <style>
        body {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title and tabs
st.title("\U0001F493 HEART DISEASE PREDICTION SYSTEM")
tab1, tab2, tab3 = st.tabs(["\U0001F50D Predict", "\U0001F4C1 Bulk Predict", "\U0001F4CA Model Info"])

# ---------------- TAB 1 ----------------
with tab1:
    st.header("Single Patient Prediction")

    # Collect user inputs
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting BP", min_value=0, max_value=300)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    modelnames = ['DTC.pkl', 'LogisticR.pkl', 'RFC.pkl', 'SVM.pkl']
    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'SVM']

    if st.button("Predict"):
        # Encode input and make predictions
        user_data = encode_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                                 resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
        st.subheader("\U0001F50E Results:")
        predictions = predict_with_all_models(user_data, modelnames)

        # Display predictions
        for algo, pred in zip(algonames, predictions):
            if pred is None:
                st.write(f"**{algo}:** Prediction Error")
            else:
                st.write(f"**{algo}:** {'\u274C Disease Detected' if pred == 1 else '\u2705 No Disease Detected'}")

# ---------------- TAB 2 ----------------
with tab2:
    st.header("Bulk Predictions via CSV Upload")

    st.markdown("""
    **Upload Guidelines:**
    - 11 columns in order:
        `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`
    - All values must be numeric. No missing or NaN values allowed.
    - Column encodings must follow:
        - `Sex: 0=Male, 1=Female`
        - `ChestPainType: 3=Typical, 0=Atypical, 1=Non-Anginal, 2=Asymptomatic`
        - `FastingBS: 1= >120 mg/dl, 0= otherwise`
        - `RestingECG: 0=Normal, 1=ST-T abnormality, 2=LVH`
        - `ExerciseAngina: 1=Yes, 0=No`
        - `ST_Slope: 0=Upsloping, 1=Flat, 2=Downsloping`
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        expected_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                         'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        # Check column validity
        if all(col in df.columns for col in expected_cols):
            try:
                model = load_model("LogisticR.pkl")
                df['Prediction'] = model.predict(df[expected_cols])
                st.success("Predictions completed.")
                st.write(df)
                st.markdown(download_link(df), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Uploaded CSV does not match the required column format.")

# ---------------- TAB 3 ----------------
with tab3:
    st.header("Model Accuracies")
    model_scores = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22
    }
    df_scores = pd.DataFrame(list(model_scores.items()), columns=['Model', 'Accuracy (%)'])
    fig = px.bar(df_scores, x='Model', y='Accuracy (%)', color='Model', text='Accuracy (%)')
    st.plotly_chart(fig, use_container_width=True)
