# Employee Income Prediction Using Machine Learning (SMOTE Enhanced)

This project aims to predict whether an individual earns more than \$50K per year based on demographic and employment-related features. The model is trained on the **UCI Adult Income Dataset** and uses multiple classification algorithms with proper data preprocessing, feature engineering, and **SMOTE** oversampling to handle class imbalance.

---

## 🔍 Problem Statement

Predict if an employee earns **>50K** or **<=50K** annually using features like:
- Age
- Education
- Occupation
- Hours-per-week
- Capital gain/loss
- Workclass, Gender, etc.

---

## 📊 Dataset Overview

- **Source**: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Total Rows**: ~32,000
- **Target Variable**: `income` (<=50K or >50K)

---

## ⚙️ Technologies & Libraries

- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost
- imbalanced-learn (`SMOTE`)
- seaborn, matplotlib
- Gradio (for deployment)

---

## ✅ Key Features

- Handled missing values (`?`)
- Applied log-transform to skewed features (`capital-gain`, `capital-loss`)
- Scaled numerical and encoded categorical data using pipelines
- **Balanced classes using SMOTE oversampling**
- Trained 6 models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - SVM
  - XGBoost
- Compared models using Accuracy, Confusion Matrix, and Classification Report

---

## ⚖️ Class Imbalance Handling

The dataset is imbalanced (~75% earn <=50K). To address this:
- **SMOTE** was used after encoding and before training
- This generates synthetic examples of the minority class (>50K)

---

## 🧪 Model Performance

| Model              | Notes                              |
|--------------------|----------|-------------------------------------|
| Logistic Regression | Good baseline, handles imbalance   |
| Random Forest       | Robust with SMOTE                  |
| **XGBoost**         | Final model                        |

> Full metrics include precision, recall, and F1-score for both classes.

---
