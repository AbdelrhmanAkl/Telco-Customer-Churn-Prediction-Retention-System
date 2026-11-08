import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard | Telco Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ design tokens
INK, MUTED, LINE, BG = "#0f1c2e", "#5b6b7f", "#dde3ea", "#f4f6f9"
TEAL, GREEN, AMBER, RED = "#0e7c86", "#1f9d63", "#d98a0b", "#d33f3f"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, select {{ font-family:'IBM Plex Sans',sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.stApp {{ background:{BG}; }}
.block-container {{ max-width:1280px; padding:2.2rem 2rem 3rem; }}
section[data-testid="stSidebar"] {{ background:#fff; border-right:1px solid {LINE}; }}
h1,h2,h3,h4 {{ color:{INK}; letter-spacing:-0.01em; }}
.brand {{ display:flex; align-items:baseline; gap:.6rem; margin-bottom:.2rem; }}
.brand b {{ font-size:2rem; color:{INK}; letter-spacing:-0.02em; }}
.brand span {{ color:{TEAL}; font-weight:600; font-size:1rem; }}
.lede {{ color:{MUTED}; font-size:1.02rem; max-width:62ch; margin-bottom:1.4rem; }}
.card {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:1.1rem 1.3rem; height:100%; }}
.kpi-label {{ color:{MUTED}; font-size:.85rem; font-weight:500; }}
.kpi-value {{ color:{INK}; font-size:1.9rem; font-weight:700; line-height:1.25; }}
.kpi-sub {{ color:{MUTED}; font-size:.8rem; }}
.verdict {{ border-radius:10px; padding:1.1rem 1.4rem; margin:.4rem 0 1.1rem; border:1px solid; }}
.verdict h3 {{ margin:0 0 .25rem; font-size:1.35rem; }}
.verdict p {{ margin:0; color:{INK}; }}
.action {{ background:#fff; border:1px solid {LINE}; border-left:4px solid {TEAL}; border-radius:8px; padding:.8rem 1.1rem; margin-bottom:.6rem; }}
.action b {{ color:{INK}; }}
.action span {{ color:{MUTED}; font-size:.9rem; }}
.stButton > button, .stFormSubmitButton > button {{ border-radius:8px; font-weight:600; }}
.stFormSubmitButton > button {{ background:{TEAL}; color:#fff; border:none; padding:.7rem 0; }}
.stFormSubmitButton > button:hover {{ background:#0a626a; color:#fff; }}
div[data-testid="stForm"] {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:1.2rem 1.4rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; }}
.foot {{ color:{MUTED}; font-size:.82rem; text-align:center; margin-top:2.5rem; }}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ model
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "telco_churn_lgbm_pipeline.pkl"
CONFIG_PATH = BASE_DIR / "artifacts" / "model_config.json"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    return model, config, explainer


model, config, explainer = load_model()
THRESHOLD = config["threshold"]
LOW = 0.20

# ------------------------------------------------------------------ helpers
LABELS = {
    "tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "avg_monthly_spend": "Average monthly spend", "num_services": "Active services",
    "num_security_services": "Security services", "num_streaming_services": "Streaming services",
    "total_services": "Total services", "is_month_to_month": "Month-to-month flag",
    "is_long_term_contract": "Long-term contract flag", "is_electronic_check": "Electronic check flag",
    "is_fiber_optic": "Fiber optic flag", "has_family": "Has family", "tenure_group": "Tenure group",
    "SeniorCitizen": "Senior citizen", "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "PaymentMethod": "Payment method",
    "PaperlessBilling": "Paperless billing", "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
}


def tier(p):
    if p < LOW:
        return "Low risk", GREEN
    if p < THRESHOLD:
        return "Medium risk", AMBER
    return "High risk", RED


def kpi(label, value, sub="", color=INK):
    return (f'<div class="card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def build_features(d):
    """Same feature engineering the model was trained with."""
    d = dict(d)
    if d["InternetService"] == "No":
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            d[c] = "No internet service"
    if d["PhoneService"] == "No":
        d["MultipleLines"] = "No phone service"

    t = d["tenure"]
    d["tenure_group"] = ("0-6 months" if t <= 6 else "7-12 months" if t <= 12 else
                         "13-24 months" if t <= 24 else "25-48 months" if t <= 48 else "49-72 months")
    d["avg_monthly_spend"] = d["TotalCharges"] / t if t > 0 else d["MonthlyCharges"]

    svc = [d[k] for k in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]]
    sec = [d[k] for k in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]]
    stm = [d["StreamingTV"], d["StreamingMovies"]]
    d["num_services"] = sum(x == "Yes" for x in svc)
    d["num_security_services"] = sum(x == "Yes" for x in sec)
    d["num_streaming_services"] = sum(x == "Yes" for x in stm)
    d["is_month_to_month"] = int(d["Contract"] == "Month-to-month")
    d["is_long_term_contract"] = int(d["Contract"] in ["One year", "Two year"])
    d["is_electronic_check"] = int(d["PaymentMethod"] == "Electronic check")
    d["is_fiber_optic"] = int(d["InternetService"] == "Fiber optic")
    d["total_services"] = d["num_services"] + d["num_security_services"] + d["num_streaming_services"]
    d["has_family"] = int(d["Partner"] == "Yes" or d["Dependents"] == "Yes")
    return d


def analyze(inputs):
    customer = pd.DataFrame([build_features(inputs)])
    prob = float(model.predict_proba(customer)[0, 1])

    # SHAP, grouped back to the original business feature (one-hot columns are summed)
    pre = model.named_steps["preprocessor"]
    x = pre.transform(customer)
    x = x.toarray() if hasattr(x, "toarray") else x
    names = list(pre.get_feature_names_out())
    sv = explainer.shap_values(pd.DataFrame(x, columns=names))
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    row = sv[0, :, 1] if sv.ndim == 3 else sv[0]

    cols = sorted(customer.columns, key=len, reverse=True)
    grouped = {}
    for n, v in zip(names, row):
        raw = n.split("__", 1)[-1]
        base = next((c for c in cols if raw == c or raw.startswith(c + "_")), raw)
        grouped[base] = grouped.get(base, 0.0) + float(v)

    drivers = []
    for base, v in grouped.items():
        val = customer[base].iloc[0] if base in customer.columns else ""
        val = f"{val:.1f}" if isinstance(val, (float, np.floating)) else val
        drivers.append({"driver": f"{LABELS.get(base, base)}: {val}" if base in customer.columns else LABELS.get(base, base),
                        "impact": v})
    drivers = pd.DataFrame(drivers).sort_values("impact", key=abs, ascending=False)
    return customer, prob, drivers


def playbook(i, prob):
    items = []
    if i["Contract"] == "Month-to-month":
        items.append(("Move to a 12-month plan", "Month-to-month customers churn most. Offer a loyalty discount for committing."))
    if i["PaymentMethod"] == "Electronic check":
        items.append(("Switch to automatic payment", "Offer a small bill credit for bank transfer or card autopay."))
    if i["InternetService"] != "No" and (i["TechSupport"] == "No" or i["OnlineSecurity"] == "No"):
        items.append(("Add a free support and security trial", "Customers without these add-ons are less attached to the service."))
    if i["InternetService"] == "Fiber optic" and i["MonthlyCharges"] >= 80:
        items.append(("Review price and service quality", "High fiber bills raise churn risk. Check speed complaints and offer a plan review."))
    if i["tenure"] <= 12:
        items.append(("Schedule a check-in call", "The first year is the most fragile. A proactive call builds loyalty."))
    if not items:
        items.append(("Keep standard engagement", "No major risk drivers found. Continue regular communication."))
    return items


# ------------------------------------------------------------------ demo presets
PRESETS = {
    "At-risk newcomer": dict(tenure=2, Contract="Month-to-month", InternetService="Fiber optic",
                             PaymentMethod="Electronic check", OnlineSecurity="No", TechSupport="No",
                             OnlineBackup="No", DeviceProtection="No", Partner="No", Dependents="No",
                             MonthlyCharges=95.0, TotalCharges=190.0),
    "Loyal customer": dict(tenure=60, Contract="Two year", InternetService="DSL",
                           PaymentMethod="Credit card (automatic)", OnlineSecurity="Yes", TechSupport="Yes",
                           OnlineBackup="Yes", DeviceProtection="Yes", Partner="Yes", Dependents="Yes",
                           MonthlyCharges=65.0, TotalCharges=3900.0),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ChurnGuard")
    st.caption("Retention intelligence for telecom teams")
    st.divider()
    st.markdown("**Model**")
    st.write("LightGBM classifier with SHAP explanations")
    st.write(f"Decision threshold: **{THRESHOLD:.0%}**")
    for k, v in (config.get("metrics") or {}).items():
        st.write(f"{k}: **{v}**")
    st.divider()
    st.markdown("**Risk tiers**")
    st.markdown(f"<span style='color:{GREEN}'>●</span> Low risk: below {LOW:.0%}", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{AMBER}'>●</span> Medium risk: {LOW:.0%} to {THRESHOLD:.0%}", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{RED}'>●</span> High risk: {THRESHOLD:.0%} and above", unsafe_allow_html=True)
    st.divider()
    st.caption("Predictions support, and do not replace, account manager judgment.")

# ------------------------------------------------------------------ header
st.markdown('<div class="brand"><b>ChurnGuard</b><span>Telco retention intelligence</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lede">Estimate how likely a customer is to leave, see exactly why, '
    'and get a ready retention plan with the revenue at stake.</div>',
    unsafe_allow_html=True,
)

pc1, pc2, _ = st.columns([1, 1, 3])
pc1.button("Load at-risk example", on_click=apply_preset, args=("At-risk newcomer",), use_container_width=True)
pc2.button("Load loyal example", on_click=apply_preset, args=("Loyal customer",), use_container_width=True)


def sel(label, opts, key):
    return st.selectbox(label, opts, key=key)


YN, YNI = ["Yes", "No"], lambda: ["Yes", "No", "No internet service"]

with st.form("customer_form"):
    t1, t2, t3 = st.tabs(["Customer", "Services", "Contract and billing"])
    with t1:
        a, b, c = st.columns(3)
        with a:
            gender = sel("Gender", ["Male", "Female"], "gender")
            partner = sel("Partner", YN, "Partner")
        with b:
            senior = sel("Senior citizen", [0, 1], "SeniorCitizen")
            dependents = sel("Dependents", YN, "Dependents")
        with c:
            tenure = st.number_input("Tenure (months)", 0, 72, 12, key="tenure")
    with t2:
        a, b, c = st.columns(3)
        with a:
            phone = sel("Phone service", YN, "PhoneService")
            lines = sel("Multiple lines", ["Yes", "No", "No phone service"], "MultipleLines")
            internet = sel("Internet service", ["DSL", "Fiber optic", "No"], "InternetService")
        with b:
            security = sel("Online security", YNI(), "OnlineSecurity")
            backup = sel("Online backup", YNI(), "OnlineBackup")
            device = sel("Device protection", YNI(), "DeviceProtection")
        with c:
            support = sel("Tech support", YNI(), "TechSupport")
            tv = sel("Streaming TV", YNI(), "StreamingTV")
            movies = sel("Streaming movies", YNI(), "StreamingMovies")
    with t3:
        a, b, c = st.columns(3)
        with a:
            contract = sel("Contract", ["Month-to-month", "One year", "Two year"], "Contract")
            paperless = sel("Paperless billing", YN, "PaperlessBilling")
        with b:
            payment = sel("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                             "Credit card (automatic)"], "PaymentMethod")
        with c:
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, key="MonthlyCharges")
            total = st.number_input("Total charges ($)", 0.0, 10000.0, 1000.0, key="TotalCharges")
    submitted = st.form_submit_button("Analyze customer", use_container_width=True)

if submitted:
    inputs = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": device, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    with st.spinner("Scoring customer..."):
        customer, prob, drivers = analyze(inputs)
    st.session_state["result"] = dict(inputs=inputs, customer=customer, prob=prob, drivers=drivers)

# ------------------------------------------------------------------ results
if "result" in st.session_state:
    r = st.session_state["result"]
    inputs, customer, prob, drivers = r["inputs"], r["customer"], r["prob"], r["drivers"]
    label, color = tier(prob)
    annual = inputs["MonthlyCharges"] * 12
    will_churn = prob >= THRESHOLD

    headline = {
        "High risk": ("Act now: this customer is likely to leave",
                      "Probability is above the decision threshold. Start the retention plan below."),
        "Medium risk": ("Watch closely: some churn risk",
                        "Below the decision threshold, but worth proactive engagement."),
        "Low risk": ("Stable: no action needed",
                     "Churn probability is low. Continue normal engagement."),
    }[label]
    st.markdown(
        f'<div class="verdict" style="border-color:{color};background:{color}14">'
        f'<h3 style="color:{color}">{headline[0]}</h3><p>{headline[1]}</p></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Churn probability", f"{prob:.1%}", f"Threshold {THRESHOLD:.0%}", color), unsafe_allow_html=True)
    k2.markdown(kpi("Risk tier", label, "Based on probability bands", color), unsafe_allow_html=True)
    k3.markdown(kpi("Annual revenue at stake", f"${annual:,.0f}", "Monthly charges x 12"), unsafe_allow_html=True)
    k4.markdown(kpi("Expected annual loss", f"${annual * prob:,.0f}", "Revenue x churn probability"), unsafe_allow_html=True)

    st.write("")
    g, s = st.columns([2, 3])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"color": INK, "size": 44}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1}, "bar": {"color": INK, "thickness": 0.22},
                   "steps": [{"range": [0, LOW * 100], "color": "#cfeadd"},
                             {"range": [LOW * 100, THRESHOLD * 100], "color": "#f6e2b9"},
                             {"range": [THRESHOLD * 100, 100], "color": "#f2c4c4"}],
                   "threshold": {"line": {"color": RED, "width": 4}, "thickness": 0.8, "value": THRESHOLD * 100}},
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          title=dict(text="Churn probability", font=dict(size=16, color=INK)))
        st.plotly_chart(fig, use_container_width=True)
    with s:
        top = drivers.head(8).iloc[::-1]
        bar = go.Figure(go.Bar(
            x=top["impact"], y=top["driver"], orientation="h",
            marker_color=[RED if v > 0 else GREEN for v in top["impact"]],
            text=[f"{v:+.2f}" for v in top["impact"]], textposition="outside",
        ))
        bar.update_layout(height=300, margin=dict(l=10, r=30, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Pushes churn risk down  |  up", zeroline=True,
                                                                     zerolinecolor=MUTED, gridcolor=LINE),
                          title=dict(text="Why this score: top drivers", font=dict(size=16, color=INK)))
        st.plotly_chart(bar, use_container_width=True)
    st.caption("Driver values are SHAP contributions in log-odds. Red raises churn risk, green lowers it. "
               "One-hot columns are combined per feature.")

    st.markdown("### Retention plan")
    if will_churn or prob >= LOW:
        for title, why in playbook(inputs, prob):
            st.markdown(f'<div class="action"><b>{title}</b><br><span>{why}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="action"><b>Keep standard engagement</b><br><span>No intervention required.</span></div>',
                    unsafe_allow_html=True)

    with st.expander("Customer details and export"):
        prof = pd.DataFrame({"Field": [LABELS.get(c, c) for c in customer.columns],
                             "Value": customer.iloc[0].astype(str).values})
        st.dataframe(prof, use_container_width=True, hide_index=True)
        report = customer.copy()
        report.insert(0, "churn_probability", round(prob, 4))
        report.insert(1, "risk_tier", label)
        report.insert(2, "prediction", "Churn" if will_churn else "No churn")
        report["annual_revenue_at_stake"] = annual
        st.download_button("Download customer report (CSV)", report.to_csv(index=False).encode("utf-8"),
                           "churnguard_customer_report.csv", "text/csv", use_container_width=True)

st.markdown('<div class="foot">ChurnGuard · LightGBM, SHAP and Streamlit</div>', unsafe_allow_html=True)
