# Telco Customer Churn Prediction and Retention System

An end-to-end machine learning system that predicts customer churn, explains the drivers behind each prediction, and turns the result into a concrete retention plan with the revenue at stake.

Live demo: https://telco-customer-churn-prediction-retention-system-m.streamlit.app/

---

## Overview

Customer churn is one of the costliest problems in telecommunications. Predicting who might leave is only the first step; a retention team also needs to know why, how much revenue is at risk, and what to do next.

This project is built as a decision-support tool that follows one flow:

```text
Prediction  ->  Risk tier  ->  Explanation  ->  Retention action
```

### What the application does

- Takes customer profile, services, and billing details as input
- Predicts the probability of churn
- Classifies the customer into a Low, Medium, or High risk tier
- Estimates annual revenue at stake and expected annual loss
- Explains the top drivers of the prediction with SHAP
- Recommends retention actions based on the customer's situation
- Exports a customer report as CSV
- Includes two built-in examples (at-risk and loyal customer) for quick demonstrations

---

## Model Performance

The production model is a LightGBM classifier inside a complete scikit-learn pipeline.

| Metric    | Score  |
| --------- | -----: |
| Accuracy  | 76.63% |
| Precision | 54.38% |
| Recall    | 75.09% |
| F1 score  | 63.08% |
| ROC-AUC   | 83.77% |
| PR-AUC    | 65.59% |

### Decision threshold

The production threshold is **0.29** instead of the default 0.50.

```text
P(churn) >= 0.29   ->  flagged as churn risk
P(churn) <  0.29   ->  not flagged
```

In retention work, missing a customer who is about to leave usually costs more than contacting one extra customer who would have stayed. A lower threshold trades some precision for higher recall, which is why recall reaches 75%.

### Risk tiers

| Tier   | Churn probability |
| ------ | ----------------- |
| Low    | below 20%         |
| Medium | 20% to below 29%  |
| High   | 29% and above     |

---

## Pipeline

```text
Customer input
      |
      v
Feature engineering
      |
      v
Preprocessing (ColumnTransformer)
  - Numeric:     median imputation, standard scaling
  - Categorical: most-frequent imputation, one-hot encoding
      |
      v
LightGBM classifier
      |
      v
Churn probability
      |
      v
Threshold (0.29) and risk tier
      |
      +--> SHAP explanation
      |
      v
Retention recommendation
```

### Engineered features

| Feature                  | Meaning                                              |
| ------------------------ | ---------------------------------------------------- |
| `tenure_group`           | Tenure bucketed into five ranges                     |
| `avg_monthly_spend`      | Total charges divided by tenure                      |
| `num_services`           | Count of active services                             |
| `num_security_services`  | Security-related add-ons in use                      |
| `num_streaming_services` | Streaming add-ons in use                             |
| `is_month_to_month`      | Month-to-month contract flag                         |
| `is_long_term_contract`  | One-year or two-year contract flag                   |
| `is_electronic_check`    | Pays by electronic check                             |
| `is_fiber_optic`         | Uses fiber optic internet                            |
| `total_services`         | Combined service count                               |
| `has_family`             | Has a partner or dependents                          |

---

## Explainability

Each prediction comes with a SHAP breakdown that shows which factors push the churn probability up and which push it down.

Contributions from one-hot encoded columns are combined back into the original business feature, and each driver is shown with the customer's actual value, for example `Contract: Month-to-month` or `Tech support: No`. This keeps the explanation readable for non-technical users.

---

## Retention Recommendations

Actions are generated from the customer's own profile. Examples:

- Month-to-month contract: offer a 12-month plan with a loyalty discount
- Electronic check: encourage automatic payment with a small bill credit
- No tech support or online security: offer a free trial of these add-ons
- High fiber optic bill: review pricing and service quality
- Tenure of 12 months or less: schedule a proactive check-in call

Low-risk customers are marked as needing no intervention.

---

## Technology Stack

| Area                    | Tools                                             |
| ----------------------- | ------------------------------------------------- |
| Machine learning        | Python, Pandas, NumPy, scikit-learn, LightGBM, Joblib |
| Explainability          | SHAP                                              |
| Application             | Streamlit, Plotly                                 |
| Deployment              | Git, GitHub, Streamlit Community Cloud            |
| Development             | Jupyter Notebook, Python virtual environment      |

---

## Project Structure

```text
Telco-Customer-Churn-Prediction-Retention-System/
|
|-- app/
|   `-- app.py
|
|-- artifacts/
|   |-- model_config.json
|   `-- telco_churn_lgbm_pipeline.pkl
|
|-- notebooks/
|   `-- telco_churn_analysis.ipynb
|
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/AbdelrhmanAkl/Telco-Customer-Churn-Prediction-Retention-System.git
cd Telco-Customer-Churn-Prediction-Retention-System
```

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The application opens in your browser.

---

## Model Artifacts

| File                                     | Purpose                                              |
| ---------------------------------------- | ---------------------------------------------------- |
| `artifacts/telco_churn_lgbm_pipeline.pkl` | Preprocessing and LightGBM classifier in one pipeline |
| `artifacts/model_config.json`            | Production threshold and optional display metrics    |

Because preprocessing is saved inside the pipeline, the app can score raw customer inputs without rebuilding any transformation at inference time.

Optionally, add a `metrics` object to `model_config.json` (for example `{"ROC-AUC": "0.84", "Recall": "0.75"}`) and the app will show it in the header.

---

## Reproducibility

The full analysis and modeling workflow is in `notebooks/telco_churn_analysis.ipynb`.

---

## Deployment

The app is deployed on Streamlit Community Cloud.

| Setting   | Value                                                          |
| --------- | -------------------------------------------------------------- |
| Repository | `AbdelrhmanAkl/Telco-Customer-Churn-Prediction-Retention-System` |
| Branch    | `main`                                                         |
| Main file | `app/app.py`                                                   |
| Python    | 3.11                                                           |

Paths to the model and configuration are resolved relative to the file, so the app runs the same way locally and in the cloud.

---

## Production-Oriented Design

- Full preprocessing pipeline saved together with the model
- Unknown categories handled with `handle_unknown="ignore"`
- Missing numeric values filled by median, missing categories by most frequent value
- Tuned decision threshold instead of the default 0.50
- Explanations grouped by business feature
- Risk tiers and revenue-at-stake estimates for business users
- Downloadable customer reports
- Deployment-safe artifact paths

---

## Author

**Abdelrahman Ahmed Akl**
AI / Machine Learning Engineer
NLP, LLMs, Generative AI, AI Agents, Machine Learning

GitHub: https://github.com/AbdelrhmanAkl
