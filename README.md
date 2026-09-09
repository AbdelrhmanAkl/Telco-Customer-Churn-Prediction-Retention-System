# 📊 Telco Customer Churn Prediction & Retention System

> An end-to-end Machine Learning system for predicting customer churn, identifying high-risk customers, explaining churn drivers, and supporting proactive customer retention decisions.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)

---

## 🚀 Live Demo

### 🌐 Try the Application

**Telco Churn AI — Live Streamlit Application**

👉 [Launch the Live Demo](https://telco-customer-churn-prediction-retention-system-m.streamlit.app/)

The deployed application allows users to:

* Enter customer information
* Predict churn probability
* Classify customer risk level
* Analyze customer risk factors
* Generate personalized retention recommendations
* Explore SHAP-based model explanations
* Download a customer prediction report

---

## 🎯 Project Overview

Customer churn is one of the most important challenges in the telecommunications industry.

Instead of simply predicting whether a customer will leave, this project is designed as a **decision-support system** that combines:

**Prediction → Risk Assessment → Explainability → Retention Action**

The system predicts the probability that a customer will churn and transforms that prediction into actionable retention insights.

### Business Objective

The main objective is to help a telecom company:

* Identify customers at high risk of churn
* Prioritize retention efforts
* Understand the factors driving churn
* Reduce unnecessary retention interventions
* Support data-driven customer relationship management

---

## 🧠 Machine Learning Approach

The final production model is a **LightGBM classifier** integrated into a complete Scikit-learn pipeline.

### End-to-End Pipeline

```text
Customer Information
        │
        ▼
Data Preprocessing
        │
        ├── Numerical Features
        │     ├── Median Imputation
        │     └── Standard Scaling
        │
        └── Categorical Features
              ├── Most-Frequent Imputation
              └── One-Hot Encoding
        │
        ▼
Feature Engineering
        │
        ▼
LightGBM Classifier
        │
        ▼
Churn Probability
        │
        ▼
Optimized Decision Threshold
        │
        ▼
Risk Classification
        │
        ├── 🟢 Low Risk
        ├── 🟡 Medium Risk
        └── 🔴 High Risk
        │
        ▼
SHAP Explainability
        │
        ▼
Retention Recommendation
```

---

## ⚙️ Feature Engineering

The system derives additional business-oriented features from the original customer attributes.

Key engineered features include:

* `tenure_group`
* `avg_monthly_spend`
* `num_services`
* `num_security_services`
* `num_streaming_services`
* `is_month_to_month`
* `is_long_term_contract`
* `is_electronic_check`
* `is_fiber_optic`
* `total_services`
* `has_family`

These features allow the model to capture customer behavior and service relationships that may not be represented directly by individual raw variables.

---

## 📈 Model Performance

The final LightGBM model was evaluated using multiple classification metrics.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **76.63%** |
| Precision | **54.38%** |
| Recall    | **75.09%** |
| F1 Score  | **63.08%** |
| ROC-AUC   | **83.77%** |
| PR-AUC    | **65.59%** |

### 🎯 Optimized Classification Threshold

The production decision threshold is:

```text
0.29
```

Instead of relying on the default `0.50` threshold, the system uses an optimized threshold of **0.29** to increase the model's ability to identify customers who are likely to churn.

This is particularly relevant for retention scenarios where **missing a potential churner can be more costly than contacting an additional low-risk customer**.

```text
P(Churn) >= 0.29  →  Churn Risk
P(Churn) <  0.29  →  No Churn
```

---

## 🔍 Explainable AI

The application integrates **SHAP (SHapley Additive exPlanations)** to make model predictions more interpretable.

Instead of only returning:

```text
Customer is likely to churn
```

the system can provide insight into **why** the customer received a particular prediction.

Examples of potential churn drivers include:

* Month-to-month contracts
* Electronic check payment
* High monthly charges
* Short customer tenure
* Lack of technical support
* Lack of online security services
* Fiber-optic internet service

The application separates factors into:

### 🔴 Factors Increasing Churn Risk

Features contributing toward a higher churn probability.

### 🟢 Factors Reducing Churn Risk

Features contributing toward a lower churn probability.

This makes the system more useful for business decision-making and model interpretation.

---

## 💡 Retention Decision Support

The system goes beyond prediction by generating retention recommendations based on the customer's risk level.

### 🔴 High Risk

Recommended actions may include:

* Personalized retention offers
* Long-term contract incentives
* Targeted discounts
* Technical support interventions
* Alternative payment options

### 🟡 Medium Risk

Recommended actions may include:

* Personalized engagement campaigns
* Loyalty benefits
* Monitoring churn probability
* Promoting longer-term contracts

### 🟢 Low Risk

Customers are considered to require no immediate intervention.

---

## 🏗️ Project Architecture

```text
                         ┌──────────────────────┐
                         │   Customer Inputs    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Preprocessing       │
                         │  ColumnTransformer   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      LightGBM        │
                         │   Churn Classifier   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Churn Probability    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Threshold = 0.29     │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐             ┌──────────────────┐
          │ Risk Assessment  │             │ SHAP Explainability│
          └────────┬─────────┘             └────────┬─────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Retention Strategy   │
                         └──────────────────────┘
```

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Joblib

### Explainable AI

* SHAP

### Application & Deployment

* Streamlit
* Git
* GitHub
* Streamlit Community Cloud

### Development

* Jupyter Notebook
* Python Virtual Environment

---

## 📁 Project Structure

```text
Telco-Customer-Churn-Prediction-Retention-System/
│
├── app/
│   └── app.py
│
├── artifacts/
│   ├── model_config.json
│   └── telco_churn_lgbm_pipeline.pkl
│
├── notebooks/
│   └── telco_churn_analysis.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/AbdelrhmanAkl/Telco-Customer-Churn-Prediction-Retention-System.git
```

Navigate to the project:

```bash
cd Telco-Customer-Churn-Prediction-Retention-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## 📊 Model Artifact

The trained production pipeline is stored in:

```text
artifacts/telco_churn_lgbm_pipeline.pkl
```

The model configuration and production threshold are stored in:

```text
artifacts/model_config.json
```

The saved pipeline contains both preprocessing and the LightGBM classifier, allowing the application to process raw customer inputs without rebuilding the preprocessing pipeline at inference time.

---

## 🧪 Reproducibility

The project includes the complete analysis notebook:

```text
notebooks/telco_churn_analysis.ipynb
```

The notebook contains the machine learning workflow used to analyze the data and develop the final model.

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment configuration:

```text
Repository:
AbdelrhmanAkl/Telco-Customer-Churn-Prediction-Retention-System

Branch:
main

Main file:
app/app.py

Python:
3.11
```

The application uses deployment-safe paths to locate the trained model and configuration files regardless of the execution directory.

---

## 🔐 Production-Oriented Design

Several decisions were made to make the project closer to a real-world ML application:

* Complete preprocessing pipeline saved with the model
* Unknown categorical values handled using `handle_unknown="ignore"`
* Missing numerical values handled using median imputation
* Missing categorical values handled using most-frequent imputation
* Optimized classification threshold
* Explainable predictions using SHAP
* Business-oriented risk categorization
* Automated retention recommendations
* Downloadable customer prediction reports
* Deployment-safe artifact paths
* Streamlit-based interactive interface

---

## 🎯 Key Takeaways

This project demonstrates an end-to-end **Machine Learning → Explainability → Business Decision** workflow.

Rather than stopping at model training, the system focuses on how a machine learning model can be transformed into an interactive business application that supports customer retention decisions.

### Core capabilities

```text
✔ Customer Churn Prediction
✔ Probability Estimation
✔ Risk Classification
✔ Threshold Optimization
✔ Feature Engineering
✔ LightGBM Classification
✔ SHAP Explainability
✔ Customer Risk Analysis
✔ Retention Recommendations
✔ Customer Reporting
✔ Streamlit Deployment
```

---

## 👨‍💻 Author

**Abdelrahman Ahmed Akl**

AI / Machine Learning Engineer
NLP • LLMs • Generative AI • AI Agents • Machine Learning

---

## ⭐ If You Find This Project Useful

Feel free to explore the repository, try the live application, and provide feedback.

---

### 📌 Project Status

**Status:** Production-ready portfolio project
**Model:** LightGBM
**Explainability:** SHAP
**Deployment:** Streamlit Community Cloud
**Python:** 3.11
