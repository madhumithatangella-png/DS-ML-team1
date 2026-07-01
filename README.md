# Contextual Predictive Maintenance using IoT Edge AI

## Project Overview

This project focuses on developing a Contextual Predictive Maintenance system that predicts machine failures before they occur by combining internal IoT sensor telemetry with contextual environmental features. The solution leverages machine learning techniques to improve maintenance planning, reduce unexpected equipment downtime, and support data-driven industrial decision-making.

The project is being developed as part of an AI & Data Science Internship and follows a structured four-week engineering roadmap.

---

## Problem Statement

Traditional predictive maintenance models rely only on machine sensor data. However, real-world machine failures are influenced by both internal operating conditions and external environmental factors.

This project builds a contextual predictive maintenance pipeline by integrating IoT telemetry with contextual features, followed by advanced machine learning techniques to accurately predict machine failures.

---

## Business Objectives

- Reduce unexpected machine failures
- Improve maintenance scheduling
- Minimize operational downtime
- Increase equipment reliability
- Develop an explainable AI-based predictive maintenance solution

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains industrial machine telemetry including:

- Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure

Additional contextual features are engineered during the project.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- SHAP
- Joblib
- Jupyter Notebook

---

# Project Workflow

```
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Signal Processing
        ↓
Feature Engineering
        ↓
Contextual Data Fusion
        ↓
SMOTE for Class Balancing
        ↓
LightGBM Classification
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Noise Sensitivity Analysis (In Progress)
```

---

# Week-wise Progress

## Week 1 – IoT Telemetry Ingestion & Signal Processing

Completed:

- Dataset loading
- Data cleaning
- Missing value analysis
- Exploratory Data Analysis
- Label Encoding
- Rolling Mean
- Rolling Standard Deviation
- Rolling Variance
- Feature Engineering
- Data preprocessing

Output:

- `week1_processed_data.csv`

---

## Week 2 – Contextual Data Fusion & Feature Engineering

Completed:

- Added Ambient Temperature
- Added Humidity
- Added Factory Load
- Added Work Shift
- Feature Encoding
- Contextual Feature Engineering
- Correlation Analysis
- Data Visualization

Output:

- `week2_processed_data.csv`

---

## Week 3 – Imbalanced Classification & LightGBM Modeling

Completed:

- Train-Test Split
- SMOTE
- LightGBM Classifier
- Model Evaluation
- Classification Report
- Confusion Matrix
- Stratified Cross Validation
- SHAP Explainability
- Model Saving

Output:

- `predictive_maintenance_model.pkl`

---

## Week 4 – Noise Sensitivity Analysis & Threshold Tuning

Current Progress (July 1)

Completed:

- Week 4 notebook created
- Required libraries imported
- Trained model loaded
- Processed dataset loaded
- Feature and target preparation
- Train-test split completed
- Data prepared for robustness testing

Upcoming Tasks:

- Gaussian Noise Injection
- Robustness Analysis
- Precision-Recall Curve
- Threshold Tuning
- Final Evaluation

---

# Project Structure

```
Contextual-Predictive-Maintenance/

│── notebooks/
│   ├── Week1_IoT_Telemetry_Signal_Processing.ipynb
│   ├── Week2_Contextual_Data_Fusion.ipynb
│   ├── Week3_Imbalanced_Classification_LightGBM_Modeling.ipynb
│   └── Week4_Noise_Sensitivity_Analysis_Threshold_Tuning.ipynb

│── data/
│   ├── ai4i2020.csv
│   ├── week1_processed_data.csv
│   └── week2_processed_data.csv

│── predictive_maintenance_model.pkl

│── images/

│── README.md
```

---

# Machine Learning Pipeline

- Data Cleaning
- Feature Engineering
- Contextual Data Fusion
- Class Balancing using SMOTE
- LightGBM Classification
- Model Evaluation
- Explainable AI using SHAP
- Noise Robustness Testing

---

# Model Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Confusion Matrix
- Stratified Cross Validation
- Precision-Recall Curve (Week 4)
- SHAP Feature Importance

---

# Current Project Status

| Phase | Status |
|--------|--------|
| Week 1 | ✅ Completed |
| Week 2 | ✅ Completed |
| Week 3 | ✅ Completed |
| Week 4 | 🚧 In Progress |

---

# Future Improvements

- Complete Noise Sensitivity Analysis
- Decision Threshold Optimization
- Model Robustness Testing
- Interactive Dashboard
- Real-time IoT Data Integration
- Model Deployment

---

# Author

**Tangella Madhumitha**

Final Year B.Tech Student

AI & Data Science Intern

---

## Repository Status

🚀 Project currently in active development.

**Progress:** Week 4 (Initial Setup Completed)

Expected Completion Date: **6 July 2026**

- Real-time machine sensor monitoring
- Streamlit-based prediction dashboard
- Cloud deployment
- Automated maintenance alert system
- Advanced model optimization techniques
