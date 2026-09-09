import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telco Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.4rem;
        font-size: 2.3rem;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .risk-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        text-align: center;
        min-height: 120px;
    }

    .risk-card-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .risk-card-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }

    .profile-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "telco_churn_lgbm_pipeline.pkl"
CONFIG_PATH = BASE_DIR / "artifacts" / "model_config.json"


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, config


model, config = load_model()
THRESHOLD = config["threshold"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Telco Churn AI")

    st.markdown("---")

    st.subheader("🤖 Model Information")

    st.write("**Algorithm:** LightGBM")
    st.write("**Task:** Binary Classification")
    st.write(
        f"**Decision Threshold:** {THRESHOLD:.0%}"
    )

    st.markdown("---")

    st.subheader("🎯 Risk Levels")

    st.write("🟢 **Low Risk**")
    st.caption("Churn probability < 20%")

    st.write("🟡 **Medium Risk**")
    st.caption(
        f"20% ≤ probability < {THRESHOLD:.0%}"
    )

    st.write("🔴 **High Risk**")
    st.caption(
        f"Probability ≥ {THRESHOLD:.0%}"
    )

    st.markdown("---")

    st.info(
        """
        This system provides AI-powered
        churn prediction and retention
        decision support.
        """
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📊 Telco Customer Churn Prediction</h1>
        <p>
        AI-Powered Customer Retention Decision Support System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    if tenure <= 6:

        tenure_group = "0-6 months"

    elif tenure <= 12:

        tenure_group = "7-12 months"

    elif tenure <= 24:

        tenure_group = "13-24 months"

    elif tenure <= 48:

        tenure_group = "25-48 months"

    else:

        tenure_group = "49-72 months"


    if tenure > 0:

        avg_monthly_spend = (
            total_charges / tenure
        )

    else:

        avg_monthly_spend = monthly_charges


    service_columns = [
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]


    num_services = sum(
        1
        for x in service_columns
        if x == "Yes"
    )


    security_columns = [
        online_security,
        online_backup,
        device_protection,
        tech_support
    ]


    num_security_services = sum(
        1
        for x in security_columns
        if x == "Yes"
    )


    streaming_columns = [
        streaming_tv,
        streaming_movies
    ]


    num_streaming_services = sum(
        1
        for x in streaming_columns
        if x == "Yes"
    )


    is_month_to_month = int(
        contract == "Month-to-month"
    )


    is_long_term_contract = int(
        contract in [
            "One year",
            "Two year"
        ]
    )


    is_electronic_check = int(
        payment_method == "Electronic check"
    )


    is_fiber_optic = int(
        internet_service == "Fiber optic"
    )


    total_services = (
        num_services
        + num_security_services
        + num_streaming_services
    )


    has_family = int(
        partner == "Yes"
        or dependents == "Yes"
    )


    # ========================================================
    # CUSTOMER DATAFRAME
    # ========================================================

    customer = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges,

        "tenure_group": tenure_group,

        "avg_monthly_spend": avg_monthly_spend,

        "num_services": num_services,

        "num_security_services": num_security_services,

        "num_streaming_services": num_streaming_services,

        "is_month_to_month": is_month_to_month,

        "is_long_term_contract": is_long_term_contract,

        "is_electronic_check": is_electronic_check,

        "is_fiber_optic": is_fiber_optic,

        "total_services": total_services,

        "has_family": has_family

    }])


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    probability = model.predict_proba(
        customer
    )[0, 1]


    prediction = int(
        probability >= THRESHOLD
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.20:

        risk_level = "Low Risk"
        risk_icon = "🟢"

    elif probability < THRESHOLD:

        risk_level = "Medium Risk"
        risk_icon = "🟡"

    else:

        risk_level = "High Risk"
        risk_icon = "🔴"


    # ========================================================
    # RESULTS HEADER
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Prediction Result</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

    k1, k2, k3, k4 = st.columns(4)


    with k1:

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="risk-card-title">
                    Churn Probability
                </div>
                <div class="risk-card-value">
                    {probability:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with k2:

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="risk-card-title">
                    Risk Level
                </div>
                <div class="risk-card-value">
                    {risk_icon} {risk_level}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with k3:

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="risk-card-title">
                    Decision Threshold
                </div>
                <div class="risk-card-value">
                    {THRESHOLD:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with k4:

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="risk-card-title">
                    Customer Tenure
                </div>
                <div class="risk-card-value">
                    {tenure} mo
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown("### 📊 Churn Probability")

    st.progress(
        float(probability)
    )

    st.caption(
        f"Churn Probability: {probability:.2%} "
        f"| Decision Threshold: {THRESHOLD:.0%} "
        f"| Prediction: "
        f"{'Churn' if prediction == 1 else 'No Churn'}"
    )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.markdown("### 🎯 Risk Assessment")


    if probability < 0.20:

        st.success(
            f"""
            🟢 **Low Risk**

            This customer currently shows a low probability
            of churn and does not require immediate
            retention intervention.
            """
        )


    elif probability < THRESHOLD:

        st.warning(
            f"""
            🟡 **Medium Risk**

            This customer is below the decision threshold,
            but shows some potential churn risk.
            """
        )


    else:

        st.error(
            f"""
            🔴 **High Risk**

            This customer is above the decision threshold
            and should be considered for retention intervention.
            """
        )


    # ========================================================
    # TWO COLUMN BUSINESS VIEW
    # ========================================================

    left, right = st.columns(2)


    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    with left:

        st.markdown("### 👤 Customer Profile")

        profile_df = pd.DataFrame({
            "Attribute": [
                "Contract",
                "Internet",
                "Monthly Charges",
                "Total Charges",
                "Payment Method",
                "Tenure",
                "Services"
            ],

            "Value": [
                contract,
                internet_service,
                f"${monthly_charges:.2f}",
                f"${total_charges:.2f}",
                payment_method,
                f"{tenure} months",
                str(total_services)
            ]
        })


        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # CUSTOMER RISK FACTORS
    # ========================================================

    with right:

        st.markdown(
            "### 🔍 Customer Risk Factors"
        )

        risk_factors = []


        if contract == "Month-to-month":

            risk_factors.append(
                "📄 Month-to-month contract"
            )


        if payment_method == "Electronic check":

            risk_factors.append(
                "💳 Electronic check payment method"
            )


        if internet_service == "Fiber optic":

            risk_factors.append(
                "🌐 Fiber optic internet service"
            )


        if tenure <= 12:

            risk_factors.append(
                "⏳ Short customer tenure"
            )


        if monthly_charges >= 80:

            risk_factors.append(
                "💰 High monthly charges"
            )


        if online_security == "No":

            risk_factors.append(
                "🔐 No online security service"
            )


        if tech_support == "No":

            risk_factors.append(
                "🛠️ No technical support service"
            )


        if risk_factors:

            for factor in risk_factors:

                st.write(
                    f"- {factor}"
                )

        else:

            st.success(
                "No major predefined churn risk factors detected."
            )


    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    st.markdown("### 💡 Recommended Action")


    if prediction == 1:

        st.error(
            """
            ### 🔴 Retention Intervention Recommended

            This customer has a churn probability above
            the decision threshold.

            **Suggested actions:**

            - 🎁 Provide a personalized retention offer
            - 📄 Encourage migration to a long-term contract
            - 💰 Consider a targeted discount or loyalty incentive
            - 🛠️ Offer technical support if applicable
            - 💳 Consider alternative payment options
            """
        )


    elif probability >= 0.20:

        st.warning(
            """
            ### 🟡 Preventive Retention Recommended

            This customer is below the decision threshold,
            but proactive engagement may help prevent
            future churn.

            **Suggested actions:**

            - 📧 Send personalized engagement campaign
            - 🎁 Offer loyalty benefits
            - 📊 Monitor churn probability
            - 📄 Consider promoting a longer-term contract
            """
        )


    else:

        st.success(
            """
            ### 🟢 No Immediate Intervention Required

            Continue normal customer engagement
            and monitoring.
            """
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🧠 Model Explainability"
    )

    st.caption(
        "SHAP shows which features contributed most to this prediction."
    )


    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        lgbm_model = model.named_steps[
            "model"
        ]


        customer_processed = (
            preprocessor.transform(customer)
        )


        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        customer_processed_df = pd.DataFrame(
            customer_processed,
            columns=feature_names
        )


        explainer = shap.TreeExplainer(
            lgbm_model
        )


        shap_values = explainer.shap_values(
            customer_processed_df
        )


        if isinstance(
            shap_values,
            list
        ):

            customer_shap_values = (
                shap_values[1][0]
            )

        else:

            customer_shap_values = (
                shap_values[0]
            )


        shap_df = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": customer_shap_values

        })


        shap_df["Absolute SHAP"] = (
            shap_df["SHAP Value"]
            .abs()
        )


        shap_df = shap_df.sort_values(
            "Absolute SHAP",
            ascending=False
        )


        # ====================================================
        # BUSINESS FRIENDLY FEATURE NAMES
        # ====================================================

        feature_name_map = {

            "numeric__tenure":
                "Customer Tenure",

            "numeric__MonthlyCharges":
                "Monthly Charges",

            "numeric__TotalCharges":
                "Total Charges",

            "numeric__avg_monthly_spend":
                "Average Monthly Spend",

            "numeric__num_services":
                "Number of Services",

            "numeric__num_security_services":
                "Security Services",

            "numeric__num_streaming_services":
                "Streaming Services",

            "numeric__is_month_to_month":
                "Month-to-Month Contract",

            "numeric__is_long_term_contract":
                "Long-Term Contract",

            "numeric__is_electronic_check":
                "Electronic Check",

            "numeric__is_fiber_optic":
                "Fiber Optic Internet",

            "numeric__total_services":
                "Total Services",

            "numeric__has_family":
                "Has Family",

            "categorical__Contract_Month-to-month":
                "Contract: Month-to-month",

            "categorical__Contract_One year":
                "Contract: One year",

            "categorical__Contract_Two year":
                "Contract: Two year",

            "categorical__InternetService_DSL":
                "Internet: DSL",

            "categorical__InternetService_Fiber optic":
                "Internet: Fiber optic",

            "categorical__InternetService_No":
                "Internet: No",

            "categorical__PaymentMethod_Electronic check":
                "Payment: Electronic check",

            "categorical__PaymentMethod_Mailed check":
                "Payment: Mailed check",

            "categorical__PaymentMethod_Bank transfer (automatic)":
                "Payment: Bank transfer",

            "categorical__PaymentMethod_Credit card (automatic)":
                "Payment: Credit card",

            "categorical__OnlineSecurity_No":
                "No Online Security",

            "categorical__OnlineSecurity_Yes":
                "Online Security",

            "categorical__OnlineBackup_No":
                "No Online Backup",

            "categorical__OnlineBackup_Yes":
                "Online Backup",

            "categorical__TechSupport_No":
                "No Technical Support",

            "categorical__TechSupport_Yes":
                "Technical Support",

            "categorical__Dependents_Yes":
                "Has Dependents",

            "categorical__Dependents_No":
                "No Dependents",

            "categorical__Partner_Yes":
                "Has Partner",

            "categorical__Partner_No":
                "No Partner"
        }


        shap_df["Feature"] = (
            shap_df["Feature"]
            .map(feature_name_map)
            .fillna(shap_df["Feature"])
        )


        # ====================================================
        # POSITIVE / NEGATIVE SHAP
        # ====================================================

        increasing = shap_df[
            shap_df["SHAP Value"] > 0
        ].head(5)


        decreasing = shap_df[
            shap_df["SHAP Value"] < 0
        ].sort_values(
            "SHAP Value"
        ).head(5)


        shap_col1, shap_col2 = st.columns(2)


        with shap_col1:

            st.markdown(
                "#### 🔴 Factors Increasing Churn Risk"
            )


            if len(increasing) > 0:

                for _, row in increasing.iterrows():

                    st.write(
                        f"**{row['Feature']}** "
                        f"({row['SHAP Value']:.4f})"
                    )

            else:

                st.write(
                    "No significant factors increasing churn risk."
                )


        with shap_col2:

            st.markdown(
                "#### 🟢 Factors Reducing Churn Risk"
            )


            if len(decreasing) > 0:

                for _, row in decreasing.iterrows():

                    st.write(
                        f"**{row['Feature']}** "
                        f"({row['SHAP Value']:.4f})"
                    )

            else:

                st.write(
                    "No significant factors reducing churn risk."
                )


        # ====================================================
        # SHAP TABLE
        # ====================================================

        with st.expander(
            "📋 View Detailed SHAP Analysis"
        ):

            display_df = shap_df.head(10).copy()

            display_df["Impact"] = (
                display_df["SHAP Value"]
                .apply(
                    lambda x:
                    "🔴 Increases Churn Risk"
                    if x > 0
                    else "🟢 Decreases Churn Risk"
                )
            )


            display_df["SHAP Value"] = (
                display_df["SHAP Value"]
                .round(4)
            )


            display_df = display_df[
                [
                    "Feature",
                    "SHAP Value",
                    "Impact"
                ]
            ]


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.warning(
            f"SHAP explanation unavailable: {e}"
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 📥 Customer Report"
    )


    report = pd.DataFrame([{

        "Churn Probability":
            round(probability, 4),

        "Risk Level":
            risk_level,

        "Decision Threshold":
            THRESHOLD,

        "Prediction":
            "Churn"
            if prediction == 1
            else "No Churn",

        "Gender":
            gender,

        "Senior Citizen":
            senior_citizen,

        "Partner":
            partner,

        "Dependents":
            dependents,

        "Tenure":
            tenure,

        "Contract":
            contract,

        "Internet Service":
            internet_service,

        "Payment Method":
            payment_method,

        "Monthly Charges":
            monthly_charges,

        "Total Charges":
            total_charges,

        "Average Monthly Spend":
            avg_monthly_spend,

        "Number of Services":
            num_services,

        "Security Services":
            num_security_services,

        "Streaming Services":
            num_streaming_services

    }])


    csv = report.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download Customer Report",
        data=csv,
        file_name="telco_churn_customer_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Telco Churn AI • LightGBM • SHAP • Streamlit

        <br>

        AI-powered customer retention decision support system

    </div>
    """,
    unsafe_allow_html=True
)