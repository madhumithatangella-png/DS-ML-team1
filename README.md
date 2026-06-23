# Predictive Maintenance - Week 3
## Model Evaluation, Validation & Explainable AI (SHAP)

## Overview

This repository contains my Week 3 contribution for the Predictive Maintenance project.

The objective of this phase is to improve the reliability and interpretability of the machine failure prediction model by performing detailed model evaluation, validation, and explainability analysis.

Predictive maintenance helps industries reduce unexpected machine failures by using machine sensor data to predict potential failures before they occur.

---

# Week 3 Objectives

The main goals of this phase were:

- Evaluate the trained machine learning model performance
- Analyze classification performance using multiple metrics
- Validate model stability using cross-validation
- Understand feature contribution using Explainable AI techniques
- Identify important factors responsible for machine failures

---

# Work Completed

## 1. Model Evaluation

The trained predictive maintenance model was evaluated using multiple performance metrics.

The evaluation includes:

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

These metrics help understand how well the model identifies machine failures.

### Confusion Matrix

A confusion matrix was generated to analyze:

- Correct failure predictions
- Incorrect failure predictions
- False alarms
- Missed failures

---

# 2. Performance Metric - Macro F1 Score

Since predictive maintenance datasets usually contain fewer failure cases compared to normal operating cases, accuracy alone is not sufficient.

Macro F1 score was used because it gives equal importance to both classes:

- Normal machine operation
- Machine failure

This helps evaluate whether the model performs well on failure detection.

---

# 3. Stratified 5-Fold Cross Validation

To ensure reliable model performance, Stratified K-Fold Cross Validation was implemented.

## Why Stratified Validation?

The dataset contains imbalanced classes, where failure cases are fewer than normal cases.

Stratified splitting ensures:

- Each fold contains similar failure/non-failure distribution
- Performance is measured consistently
- Model reliability is improved

The model performance was evaluated across 5 different folds and the average F1 score was calculated.

---

# 4. Explainable AI using SHAP

To improve model transparency, SHAP (SHapley Additive exPlanations) was implemented.

SHAP helps answer:

"Why did the model predict a machine failure?"

Instead of only providing predictions, SHAP explains the contribution of individual features.

---

# SHAP Analysis Provides:

- Important features influencing failure prediction
- Positive and negative impact of features
- Individual prediction explanations

The analysis helps identify important machine parameters such as:

- Tool wear
- Torque
- Temperature-related features
- Process measurements

---

# Project Structure
