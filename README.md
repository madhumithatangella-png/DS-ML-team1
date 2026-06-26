# AI4I Predictive Maintenance - Data Preprocessing & Feature Engineering

## Project Overview

This project focuses on preparing the AI4I Predictive Maintenance Dataset for machine learning by performing data cleaning, exploratory data analysis (EDA), and feature engineering. The main objective is to improve data quality, understand important patterns, and create meaningful features that enhance predictive model performance for machine failure prediction.

---

# Week 1: Data Cleaning, Exploratory Data Analysis, and Baseline Modeling

During the first week, the primary focus was on understanding and preprocessing the dataset.

## Tasks Performed

### 1. Data Exploration

* Loaded the AI4I 2020 Predictive Maintenance dataset using Pandas.
* Examined dataset dimensions, column names, and data types.
* Analyzed the overall structure of the dataset using `df.info()` and `df.shape()`.

### 2. Data Cleaning

* Checked for missing values across all features.
* Removed unnecessary columns such as:

  * `UDI`
  * `Product ID`
* Verified data consistency and ensured the dataset was ready for modeling.

### 3. Categorical Data Encoding

* Converted the categorical feature `Type` into numerical format using Label Encoding.
* Prepared the dataset for machine learning algorithms that require numerical inputs.

### 4. Exploratory Data Analysis (EDA)

* Analyzed the distribution of the target variable (`Machine failure`).
* Created count plots to visualize machine failure frequency.
* Generated a correlation heatmap to understand relationships between numerical features.
* Identified important patterns and dependencies among variables.

### 5. Baseline Machine Learning Model

* Split the dataset into training and testing sets.
* Trained a Random Forest Classifier as a baseline model.
* Evaluated model performance using:

  * Accuracy Score
  * Classification Report
  * Confusion Matrix

### 6. Feature Importance Analysis

* Visualized feature importance scores obtained from the Random Forest model.
* Identified the most influential variables contributing to machine failure prediction.

---

# Week 2: Feature Engineering and Model Improvement

The second week focused on creating new features to enhance predictive performance and provide additional insights.

## Tasks Performed

### 1. Dataset Preparation

* Continued preprocessing by removing irrelevant columns.
* Applied Label Encoding to categorical features.

### 2. Feature Engineering

Created new domain-specific features to capture hidden relationships within the data.

#### Temperature Difference

Calculated the difference between process temperature and air temperature.

```python
Temp_Difference = Process Temperature - Air Temperature
```

This feature helps capture abnormal thermal behavior in machines.

#### Power Index

Generated a feature representing machine power characteristics.

```python
Power_Index = Rotational Speed × Torque
```

This feature reflects the overall operational load on the machine.

#### Wear-Torque Interaction

Created an interaction feature combining tool wear and torque.

```python
Wear_Torque = Tool Wear × Torque
```

This feature represents the combined effect of mechanical stress and tool degradation.

### 3. Model Training with Engineered Features

* Trained an updated Random Forest model using the newly engineered features.
* Compared performance with the baseline model.

### 4. Model Evaluation

Evaluated the enhanced model using:

* Accuracy Score
* Classification Report
* Feature Importance Analysis

### 5. Feature Importance Visualization

* Analyzed how newly engineered features contributed to prediction performance.
* Visualized feature importance through bar charts.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Project Goal

The goal of this project is to build a robust data preprocessing pipeline and improve predictive maintenance models by leveraging effective feature engineering techniques for machine failure prediction.
