import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧")

st.title("🔧 Predictive Maintenance")
st.write("Enter the machine details below to predict machine failure.")

# Load model
model = joblib.load("predictive_maintenance_model.pkl")

# ---------------- Session History ----------------

history_features = ["air", "process", "rpm", "torque", "wear"]

for feature in history_features:
    if feature not in st.session_state:
        st.session_state[feature] = []


def update_history(name, value):
    st.session_state[name].append(value)

    if len(st.session_state[name]) > 5:
        st.session_state[name].pop(0)


def rolling_stats(values):
    values = np.array(values)

    return (
        values.mean(),
        values.std(),
        values.var()
    )


# ---------------- Inputs ----------------

machine_type = st.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)

type_map = {
    "H": 0,
    "L": 1,
    "M": 2
}

Type = type_map[machine_type]

air_temp = st.number_input(
    "Air Temperature (K)",
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature (K)",
    value=310.0
)

rpm = st.number_input(
    "Rotational Speed (RPM)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    value=120
)

ambient_temp = st.number_input(
    "Ambient Temperature (°C)",
    value=30.0
)

humidity = st.slider(
    "Humidity",
    40,
    90,
    60
)

factory_load = st.slider(
    "Factory Load",
    0.50,
    1.00,
    0.75
)

shift_name = st.selectbox(
    "Shift",
    ["Morning", "Evening", "Night"]
)

shift_map = {
    "Morning": 1,
    "Evening": 0,
    "Night": 2
}

Shift = shift_map[shift_name]

predict = st.button("Predict")

if predict:

    # Update history
    update_history("air", air_temp)
    update_history("process", process_temp)
    update_history("rpm", rpm)
    update_history("torque", torque)
    update_history("wear", tool_wear)

    # Rolling Features

    air_mean, air_std, air_var = rolling_stats(st.session_state["air"])

    process_mean, process_std, process_var = rolling_stats(
        st.session_state["process"]
    )

    rpm_mean, rpm_std, rpm_var = rolling_stats(
        st.session_state["rpm"]
    )

    torque_mean, torque_std, torque_var = rolling_stats(
        st.session_state["torque"]
    )

    wear_mean, wear_std, wear_var = rolling_stats(
        st.session_state["wear"]
    )
    # ---------------- Engineered Features ----------------

    temp_difference = process_temp - air_temp

    power_index = torque * rpm

    wear_torque = tool_wear * torque

    temp_ratio = process_temp / air_temp

    rpm_per_torque = rpm / (torque + 1)

    heat_stress = ambient_temp * process_temp

    load_torque = factory_load * torque

    humidity_wear = humidity * tool_wear

    temp_load = ambient_temp * factory_load

    # ---------------- Create Model Input ----------------

    input_data = pd.DataFrame({

        "Type": [Type],

        "Air_Temperature": [air_temp],
        "Process_Temperature": [process_temp],
        "Rotational_Speed": [rpm],
        "Torque": [torque],
        "Tool_Wear": [tool_wear],

        "Torque_RollingMean": [torque_mean],
        "Torque_RollingStd": [torque_std],
        "Torque_RollingVar": [torque_var],

        "AirTemp_RollingMean": [air_mean],
        "AirTemp_RollingStd": [air_std],
        "AirTemp_RollingVar": [air_var],

        "ProcessTemp_RollingMean": [process_mean],
        "ProcessTemp_RollingStd": [process_std],
        "ProcessTemp_RollingVar": [process_var],

        "RPM_RollingMean": [rpm_mean],
        "RPM_RollingStd": [rpm_std],
        "RPM_RollingVar": [rpm_var],

        "ToolWear_RollingMean": [wear_mean],
        "ToolWear_RollingStd": [wear_std],
        "ToolWear_RollingVar": [wear_var],

        "Temp_Difference": [temp_difference],
        "Power_Index": [power_index],
        "Wear_Torque": [wear_torque],
        "Temp_Ratio": [temp_ratio],
        "RPM_per_Torque": [rpm_per_torque],

        "Ambient_Temperature": [ambient_temp],
        "Humidity": [humidity],
        "Factory_Load": [factory_load],
        "Shift": [Shift],

        "Heat_Stress": [heat_stress],
        "Load_Torque": [load_torque],
        "Humidity_Wear": [humidity_wear],
        "Temp_Load": [temp_load]

    })

    # Ensure the feature order matches the model
    input_data = input_data[[
        'Type',
        'Air_Temperature',
        'Process_Temperature',
        'Rotational_Speed',
        'Torque',
        'Tool_Wear',
        'Torque_RollingMean',
        'Torque_RollingStd',
        'Torque_RollingVar',
        'AirTemp_RollingMean',
        'AirTemp_RollingStd',
        'AirTemp_RollingVar',
        'ProcessTemp_RollingMean',
        'ProcessTemp_RollingStd',
        'ProcessTemp_RollingVar',
        'RPM_RollingMean',
        'RPM_RollingStd',
        'RPM_RollingVar',
        'ToolWear_RollingMean',
        'ToolWear_RollingStd',
        'ToolWear_RollingVar',
        'Temp_Difference',
        'Power_Index',
        'Wear_Torque',
        'Temp_Ratio',
        'RPM_per_Torque',
        'Ambient_Temperature',
        'Humidity',
        'Factory_Load',
        'Shift',
        'Heat_Stress',
        'Load_Torque',
        'Humidity_Wear',
        'Temp_Load'
    ]]

    # ---------------- Prediction ----------------

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
        st.metric("Failure Probability", f"{probability*100:.2f}%")
        st.progress(int(probability * 100))
    else:
        st.success("✅ Machine is Healthy")
        st.metric("Failure Probability", f"{probability*100:.2f}%")
        st.progress(int((1 - probability) * 100))

    st.info(
        "This prediction is generated using a LightGBM model trained with contextual, engineered, and rolling features."
    )
