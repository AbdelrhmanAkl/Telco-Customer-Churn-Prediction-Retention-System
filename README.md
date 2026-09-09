# Telco Customer Churn Prediction & Retention System

An AI-powered customer churn prediction and retention decision-support system built using Machine Learning, LightGBM, SHAP, and Streamlit.

The system predicts the probability of customer churn, classifies customers based on a business decision threshold, identifies major churn risk factors, and provides actionable retention recommendations.

---

## Project Overview

Customer churn is a major challenge for telecom companies.

This project uses machine learning to:

- Predict customer churn probability
- Identify high-risk customers
- Apply a business-optimized decision threshold
- Explain model predictions using SHAP
- Identify customer-specific churn risk factors
- Recommend retention actions
- Provide an interactive production-style Streamlit dashboard

---

## Machine Learning Model

The project uses a LightGBM classification pipeline with preprocessing and feature engineering.

### Model

- LightGBM
- Scikit-learn Pipeline
- ColumnTransformer
- One-Hot Encoding
- Numerical preprocessing

### Evaluation

The final model achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 77% |
| Churn Precision | 54% |
| Churn Recall | 75% |
| Churn F1-Score | 63% |
| Macro F1 | 73% |

The model prioritizes identifying churners, making recall for the Churn class particularly important for retention use cases.

---

## Business Decision Threshold

Instead of using the default 50% classification threshold, the system uses a customized decision threshold of:

**29%**

This threshold allows the business to identify potentially risky customers earlier and support proactive retention strategies.

### Risk Levels

| Churn Probability | Risk Level |
|---|---|
| < 20% | Low Risk |
| 20% – 29% | Medium Risk |
| ≥ 29% | High Risk |

---

## Explainable AI

The system uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

For each customer, the dashboard displays:

- Features increasing churn risk
- Features decreasing churn risk
- SHAP values
- Customer-specific risk factors

Example:

### Factors Increasing Churn Risk

- Month-to-month contract
- Short tenure
- Electronic check payment
- High average monthly spending

### Factors Reducing Churn Risk

- Long-term contract
- Security services
- Dependents
- Technical support

---

## Feature Engineering

Several business-oriented features were created, including:

- `tenure_group`
- `avg_monthly_spend`
- `num_services`
- `num_security_services`
- `num_streaming_services`
- `is_month_to_month`
- `is_long_term_contract`
- `is_electronic_check`
- `is_fiber_optic`
- `total_services`
- `has_family`

These features help capture customer behavior and churn-related patterns.

---

## Streamlit Dashboard

The project includes an interactive Streamlit application.

The dashboard allows users to enter customer information and receive:

### Prediction

- Churn probability
- Risk level
- Decision threshold
- Churn prediction

### Customer Profile

- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Customer tenure
- Number of services

### Risk Factors

The system automatically identifies predefined churn risk factors.

### Retention Recommendations

Depending on the customer's risk level, the system provides recommendations such as:

- Personalized retention offers
- Loyalty incentives
- Long-term contract promotion
- Technical support
- Alternative payment options
- Customer engagement campaigns

---

## Project Structure

```text
Telco Customer Churn Prediction & Retention System/
│
├── app/
│   └── app.py
│
├── artifacts/
│   ├── telco_churn_lgbm_pipeline.pkl
│   └── model_config.json
│
├── notebooks/
│   └── ...
│
├── data/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md