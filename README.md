# 🔧 Contextual Predictive Maintenance using IoT Edge AI

An end-to-end **AI-powered Predictive Maintenance System** that predicts industrial machine failures by combining **IoT sensor telemetry**, **contextual environmental data**, **feature engineering**, and **LightGBM**. The project includes an interactive **Streamlit web application** for real-time machine failure prediction.

---

# 🚀 Live Demo

**Streamlit Application**

[AI-Based Predictive Maintenance App](https://predictive-maintainance-testing.streamlit.app/?utm_source=chatgpt.com)

---

# 📌 Project Overview

Unexpected machine failures can result in production downtime, increased maintenance costs, and reduced operational efficiency.

This project develops a **Contextual Predictive Maintenance** system that predicts machine failures before they occur by integrating:

* Industrial IoT sensor telemetry
* Contextual environmental information
* Engineered machine features
* Rolling statistical features

The final model is deployed as an interactive Streamlit application that enables real-time machine failure prediction.

---

# 🎯 Problem Statement

Traditional predictive maintenance systems primarily rely on machine sensor data. However, machine failures are also influenced by external environmental and operational conditions.

This project addresses this limitation by incorporating contextual features alongside sensor telemetry, resulting in a more robust and intelligent predictive maintenance solution.

---

# 🎯 Business Objectives

* Reduce unexpected machine failures
* Improve maintenance scheduling
* Minimize operational downtime
* Increase equipment reliability
* Support data-driven maintenance decisions
* Develop an explainable AI-based predictive maintenance system

---

# 📊 Dataset

### AI4I 2020 Predictive Maintenance Dataset

The dataset contains industrial machine telemetry, including:

* Machine Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Machine Failure

Additional contextual and engineered features are created during preprocessing.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* LightGBM
* SHAP
* Streamlit
* Joblib
* Jupyter Notebook
* Git & GitHub

---

# ⚙️ Project Workflow

```
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Rolling Feature Generation
        ↓
Feature Engineering
        ↓
Contextual Data Fusion
        ↓
SMOTE Class Balancing
        ↓
LightGBM Classification
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Noise Sensitivity Analysis
        ↓
Streamlit Deployment
```

---

# 📅 Week-wise Progress

## ✅ Week 1 – IoT Telemetry Processing

Completed:

* Dataset Loading
* Data Cleaning
* Missing Value Analysis
* Exploratory Data Analysis
* Label Encoding
* Rolling Mean
* Rolling Standard Deviation
* Rolling Variance
* Feature Engineering
* Data Preprocessing

Output:

* `week1_processed_data.csv`

---

## ✅ Week 2 – Contextual Data Fusion

Completed:

* Ambient Temperature
* Humidity
* Factory Load
* Work Shift
* Feature Encoding
* Contextual Feature Engineering
* Correlation Analysis
* Data Visualization

Output:

* `week2_processed_data.csv`

---

## ✅ Week 3 – LightGBM Modeling & Explainability

Completed:

* Train-Test Split
* SMOTE
* LightGBM Classifier
* Model Evaluation
* Classification Report
* Confusion Matrix
* Stratified Cross Validation
* SHAP Explainability
* Model Serialization

Output:

* `predictive_maintenance_model.pkl`

---

## ✅ Week 4 – Deployment & Robustness Analysis

Completed:

* Noise Sensitivity Analysis
* Threshold Tuning
* Model Robustness Testing
* Streamlit Dashboard Development
* Interactive Prediction Interface
* End-to-End Deployment

Output:

* Streamlit Web Application

---

# 🤖 Machine Learning Pipeline

The model is trained using **34 features**.

## Operational Features

* Machine Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

---

## Rolling Statistical Features

For each operational feature:

* Rolling Mean
* Rolling Standard Deviation
* Rolling Variance

Generated for:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

---

## Engineered Features

* Temperature Difference
* Power Index
* Wear-Torque Interaction
* Temperature Ratio
* RPM per Torque

---

## Contextual Features

* Ambient Temperature
* Humidity
* Factory Load
* Shift
* Heat Stress
* Load Torque
* Humidity Wear
* Temperature Load

---

# 🌐 Streamlit Application

The project includes an interactive Streamlit dashboard for real-time machine failure prediction.

### User Inputs

* Machine Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Ambient Temperature
* Humidity
* Factory Load
* Shift

The application automatically generates:

* Rolling Features
* Engineered Features
* Contextual Features

before sending them to the trained LightGBM model.

The dashboard displays:

* Machine Status
* Failure Probability
* Real-Time Prediction

---

# 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Stratified Cross Validation
* SHAP Feature Importance
* Noise Sensitivity Analysis

---

# 📂 Project Structure

```
Contextual-Predictive-Maintenance/

│── notebooks/
│   ├── Week1_IoT_Telemetry_Signal_Processing.ipynb
│   ├── Week2_Contextual_Data_Fusion.ipynb
│   ├── Week3_LightGBM_Modeling_SHAP.ipynb
│   └── Week4_Noise_Sensitivity_Analysis.ipynb
│
│── data/
│
│── app.py
│── predictive_maintenance_model.pkl
│── requirements.txt
│── README.md
│
└── images/
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/madhumithatangella-png/DS-ML-team1.git
```

Navigate to the project directory

```bash
cd DS-ML-team1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🚀 Future Improvements

* Real-time IoT Sensor Integration
* MQTT-based Live Data Streaming
* Remaining Useful Life (RUL) Prediction
* Automated Maintenance Alerts
* Docker Containerization
* CI/CD Pipeline
* Cloud-Native Deployment
* Advanced Explainability Dashboard using SHAP

---

# 📈 Project Status

| Phase               | Status      |
| ------------------- | ----------- |
| Week 1              | ✅ Completed |
| Week 2              | ✅ Completed |
| Week 3              | ✅ Completed |
| Week 4              | ✅ Completed |
| Streamlit Dashboard | ✅ Completed |
| Model Deployment    | ✅ Completed |

---

# 👩‍💻 Author

## Tangella Madhumitha

Final Year B.Tech Computer Science Engineering Student

AI & Data Science Enthusiast

### Connect with Me

**GitHub:** https://github.com/madhumitha15-git

**Live Demo:** [Streamlit Application](https://predictive-maintainance-testing.streamlit.app/?utm_source=chatgpt.com)

---

## ⭐ Support

If you found this project useful, consider giving the repository a **⭐ Star** on GitHub.

Your support helps showcase the project and encourages further development.
