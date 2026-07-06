import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("MODEL/predictive_maintenance_model.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.title{
    text-align:center;
    font-size:40px;
    color:#0066cc;
    font-weight:bold;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}
.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='title'>🔧 Contextual Predictive Maintenance using IoT Edge AI</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Machine Failure Prediction Dashboard</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🤖 Prediction",
        "📊 Model Information",
        "👩‍💻 About"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page=="🏠 Home":

    st.header("Project Overview")

    st.write("""
This project predicts whether an industrial machine is likely to fail using Machine Learning.

### Objectives
- Predict machine failures
- Reduce maintenance cost
- Improve machine reliability
- Enable predictive maintenance

### Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- SMOTE
- Streamlit
- SHAP

### Dataset Features
- Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
""")

# ==========================================================
# PREDICTION
# ==========================================================

elif page=="🤖 Prediction":

    st.header("Machine Parameters")

    type_value = st.selectbox(
        "Machine Type",
        ["L","M","H"]
    )

    type_dict = {
        "L":0,
        "M":1,
        "H":2
    }

    air_temp = st.slider(
        "Air Temperature (K)",
        290.0,
        310.0,
        298.0
    )

    process_temp = st.slider(
        "Process Temperature (K)",
        300.0,
        320.0,
        308.0
    )

    rpm = st.slider(
        "Rotational Speed (RPM)",
        1000,
        3000,
        1500
    )

    torque = st.slider(
        "Torque (Nm)",
        0.0,
        100.0,
        40.0
    )

    tool_wear = st.slider(
        "Tool Wear (min)",
        0,
        300,
        20
    )

    if st.button("🔍 Predict Machine Failure"):

        input_data = pd.DataFrame({

            "Type":[type_dict[type_value]],
            "Air_temperature_K":[air_temp],
            "Process_temperature_K":[process_temp],
            "Rotational_speed_rpm":[rpm],
            "Torque_Nm":[torque],
            "Tool_wear_min":[tool_wear]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.write("")

        if prediction==1:

            st.error("⚠ Machine Failure Likely")

        else:

            st.success("✅ Machine is Healthy")

        st.subheader("Failure Probability")

        st.progress(float(probability))

        st.write(f"### {probability*100:.2f}%")

# ==========================================================
# MODEL
# ==========================================================

elif page=="📊 Model Information":

    st.header("Model Information")

    st.metric(
        "Algorithm",
        "LightGBM"
    )

    st.metric(
        "Class Balancing",
        "SMOTE"
    )

    st.metric(
        "Noise Robustness",
        "Completed"
    )

    st.metric(
        "Threshold Tuning",
        "Completed"
    )

    st.metric(
        "Explainability",
        "SHAP"
    )

    st.write("")

    st.info("""
The model was trained using the AI4I Predictive Maintenance Dataset.

Major Steps:

✔ Data Cleaning

✔ Label Encoding

✔ Train-Test Split

✔ SMOTE

✔ LightGBM Training

✔ Threshold Optimization

✔ Noise Sensitivity Analysis

✔ SHAP Explainability
""")

# ==========================================================
# ABOUT
# ==========================================================

else:

    st.header("About Developer")

    st.write("""
### Tangella Madhumitha

B.Tech Student

Data Science | Machine Learning | AI

#### Project

Contextual Predictive Maintenance using IoT Edge AI

#### Internship

INFOTACT Solutions

#### Tools

- Python
- Pandas
- NumPy
- Streamlit
- LightGBM
- SHAP
- GitHub

""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
"<div class='footer'>Developed by Tangella Madhumitha | INFOTACT Internship 2026</div>",
unsafe_allow_html=True
)