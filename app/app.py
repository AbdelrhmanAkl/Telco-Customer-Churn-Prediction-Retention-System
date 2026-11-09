import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="ChurnGuard", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

INK, MUTED, LINE, BG = "#111d2e", "#66758a", "#e3e8ee", "#f5f7fa"
TEAL, GREEN, AMBER, RED = "#0e7c86", "#1f9d63", "#d98a0b", "#d33f3f"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], button, input, select {{ font-family:'IBM Plex Sans',sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"], section[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{ display:none !important; }}
.stApp {{ background:{BG}; }}
.block-container {{ max-width:1240px; padding:1.6rem 2rem 2.5rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-color:{LINE}; border-radius:12px; }}
div[data-testid="stForm"] {{ border:none; padding:0; }}
.top {{ display:flex; align-items:center; gap:.8rem; margin-bottom:1.4rem; }}
.logo {{ width:38px; height:38px; border-radius:10px; background:{TEAL}; color:#fff; font-weight:700;
        display:flex; align-items:center; justify-content:center; font-size:1.2rem; }}
.name {{ color:{INK}; font-weight:700; font-size:1.25rem; line-height:1.1; }}
.tag {{ color:{MUTED}; font-size:.85rem; }}
.chips {{ margin-left:auto; display:flex; gap:.5rem; flex-wrap:wrap; }}
.chip {{ background:#fff; border:1px solid {LINE}; color:{MUTED}; border-radius:999px; padding:.25rem .75rem; font-size:.8rem; }}
.sec {{ color:{INK}; font-weight:600; font-size:1.05rem; margin:.1rem 0 .6rem; }}
.pill {{ display:inline-block; border-radius:999px; padding:.2rem .8rem; font-weight:600; font-size:.85rem; }}
.head {{ color:{INK}; font-weight:700; font-size:1.45rem; line-height:1.25; margin:.7rem 0 .3rem; }}
.sub {{ color:{MUTED}; font-size:.95rem; margin-bottom:1rem; }}
.stat {{ border-top:1px solid {LINE}; padding:.6rem 0; display:flex; justify-content:space-between; }}
.stat span {{ color:{MUTED}; font-size:.9rem; }} .stat b {{ color:{INK}; font-size:1.05rem; }}
.action {{ border-left:3px solid {TEAL}; padding:.2rem 0 .2rem .9rem; margin-bottom:.8rem; }}
.action b {{ color:{INK}; }} .action div {{ color:{MUTED}; font-size:.9rem; }}
.legend {{ color:{MUTED}; font-size:.78rem; text-align:center; }}
.empty {{ text-align:center; padding:5rem 1rem; color:{MUTED}; }}
.empty b {{ display:block; color:{INK}; font-size:1.2rem; margin-bottom:.4rem; }}
.stFormSubmitButton > button {{ background:{TEAL}; color:#fff; border:none; border-radius:8px; font-weight:600; padding:.65rem 0; }}
.stFormSubmitButton > button:hover {{ background:#0a626a; color:#fff; }}
.stButton > button {{ border-radius:8px; font-weight:500; }}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ model
BASE_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "artifacts" / "telco_churn_lgbm_pipeline.pkl")
    with open(BASE_DIR / "artifacts" / "model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config, shap.TreeExplainer(model.named_steps["model"])


model, config, explainer = load_model()
THRESHOLD, LOW = config["threshold"], 0.20

DEFAULTS = dict(gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=12,
                PhoneService="Yes", MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes",
                OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes", StreamingTV="Yes",
                StreamingMovies="Yes", Contract="Month-to-month", PaperlessBilling="Yes",
                PaymentMethod="Electronic check", MonthlyCharges=70.0, TotalCharges=1000.0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

LABELS = {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
          "avg_monthly_spend": "Avg monthly spend", "num_services": "Active services",
          "num_security_services": "Security services", "num_streaming_services": "Streaming services",
          "total_services": "Total services", "is_month_to_month": "Month-to-month",
          "is_long_term_contract": "Long-term contract", "is_electronic_check": "Electronic check",
          "is_fiber_optic": "Fiber optic", "has_family": "Has family", "tenure_group": "Tenure group",
          "SeniorCitizen": "Senior citizen", "InternetService": "Internet", "OnlineSecurity": "Online security",
          "OnlineBackup": "Online backup", "DeviceProtection": "Device protection", "TechSupport": "Tech support",
          "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "PaymentMethod": "Payment",
          "PaperlessBilling": "Paperless billing", "PhoneService": "Phone", "MultipleLines": "Multiple lines"}


def tier(p):
    if p < LOW:
        return "Low risk", GREEN
    return ("Medium risk", AMBER) if p < THRESHOLD else ("High risk", RED)


def build_features(d):
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
    d["num_services"] = sum(x == "Yes" for x in svc)
    d["num_security_services"] = sum(d[k] == "Yes" for k in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"])
    d["num_streaming_services"] = sum(d[k] == "Yes" for k in ["StreamingTV", "StreamingMovies"])
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
    pre = model.named_steps["preprocessor"]
    x = pre.transform(customer)
    x = x.toarray() if hasattr(x, "toarray") else x
    names = list(pre.get_feature_names_out())
    sv = explainer.shap_values(pd.DataFrame(x, columns=names))
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    row = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    cols = sorted(customer.columns, key=len, reverse=True)
    grouped = {}
    for n, v in zip(names, row):
        raw = n.split("__", 1)[-1]
        base = next((c for c in cols if raw == c or raw.startswith(c + "_")), raw)
        grouped[base] = grouped.get(base, 0.0) + float(v)
    rows = []
    for base, v in grouped.items():
        if base in customer.columns:
            val = customer[base].iloc[0]
            val = f"{val:.0f}" if isinstance(val, (float, np.floating)) else val
            rows.append({"driver": f"{LABELS.get(base, base)}: {val}", "impact": v})
    drivers = pd.DataFrame(rows).sort_values("impact", key=abs, ascending=False)
    return dict(inputs=inputs, customer=customer, prob=prob, drivers=drivers)


def playbook(i):
    out = []
    if i["Contract"] == "Month-to-month":
        out.append(("Offer a 12-month plan", "Month-to-month customers churn most. Reward commitment with a loyalty discount."))
    if i["PaymentMethod"] == "Electronic check":
        out.append(("Move to automatic payment", "Offer a small bill credit for bank transfer or card autopay."))
    if i["InternetService"] != "No" and (i["TechSupport"] == "No" or i["OnlineSecurity"] == "No"):
        out.append(("Add a free support and security trial", "Customers without these add-ons are less attached to the service."))
    if i["InternetService"] == "Fiber optic" and i["MonthlyCharges"] >= 80:
        out.append(("Review price and service quality", "High fiber bills raise churn risk. Check speed complaints first."))
    if i["tenure"] <= 12:
        out.append(("Schedule a check-in call", "The first year is the most fragile. A proactive call builds loyalty."))
    return out or [("Keep standard engagement", "No major risk drivers found.")]


def run(inputs):
    st.session_state["result"] = analyze(inputs)


PRESETS = {
    "risk": dict(tenure=2, Contract="Month-to-month", InternetService="Fiber optic", PaymentMethod="Electronic check",
                 OnlineSecurity="No", TechSupport="No", OnlineBackup="No", DeviceProtection="No", Partner="No",
                 Dependents="No", MonthlyCharges=95.0, TotalCharges=190.0),
    "loyal": dict(tenure=60, Contract="Two year", InternetService="DSL", PaymentMethod="Credit card (automatic)",
                  OnlineSecurity="Yes", TechSupport="Yes", OnlineBackup="Yes", DeviceProtection="Yes", Partner="Yes",
                  Dependents="Yes", MonthlyCharges=65.0, TotalCharges=3900.0),
}


def load_example(name):
    st.session_state.update(PRESETS[name])
    run({k: st.session_state[k] for k in DEFAULTS})


# ------------------------------------------------------------------ header
chips = [f'<span class="chip">LightGBM</span>', '<span class="chip">SHAP explainability</span>',
         f'<span class="chip">Threshold {THRESHOLD:.0%}</span>']
chips += [f'<span class="chip">{k}: {v}</span>' for k, v in (config.get("metrics") or {}).items()]
st.markdown(f'<div class="top"><div class="logo">C</div><div><div class="name">ChurnGuard</div>'
            f'<div class="tag">Customer retention intelligence</div></div><div class="chips">{"".join(chips)}</div></div>',
            unsafe_allow_html=True)

left, right = st.columns([4, 7], gap="large")

# ------------------------------------------------------------------ input panel
YN = ["Yes", "No"]
YNI = ["Yes", "No", "No internet service"]
with left:
    with st.container(border=True):
        st.markdown('<div class="sec">Customer</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.button("At-risk example", on_click=load_example, args=("risk",), use_container_width=True)
        b2.button("Loyal example", on_click=load_example, args=("loyal",), use_container_width=True)
        with st.form("customer_form", border=False):
            t1, t2, t3 = st.tabs(["Profile", "Services", "Billing"])
            with t1:
                a, b = st.columns(2)
                a.selectbox("Gender", ["Male", "Female"], key="gender")
                b.selectbox("Senior citizen", [0, 1], key="SeniorCitizen")
                a.selectbox("Partner", YN, key="Partner")
                b.selectbox("Dependents", YN, key="Dependents")
                st.number_input("Tenure (months)", min_value=0, max_value=72, key="tenure")
            with t2:
                a, b = st.columns(2)
                a.selectbox("Phone service", YN, key="PhoneService")
                b.selectbox("Multiple lines", ["Yes", "No", "No phone service"], key="MultipleLines")
                a.selectbox("Internet", ["DSL", "Fiber optic", "No"], key="InternetService")
                b.selectbox("Online security", YNI, key="OnlineSecurity")
                a.selectbox("Online backup", YNI, key="OnlineBackup")
                b.selectbox("Device protection", YNI, key="DeviceProtection")
                a.selectbox("Tech support", YNI, key="TechSupport")
                b.selectbox("Streaming TV", YNI, key="StreamingTV")
                a.selectbox("Streaming movies", YNI, key="StreamingMovies")
            with t3:
                st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
                st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                "Credit card (automatic)"], key="PaymentMethod")
                st.selectbox("Paperless billing", YN, key="PaperlessBilling")
                a, b = st.columns(2)
                a.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, step=1.0, key="MonthlyCharges")
                b.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, step=10.0, key="TotalCharges")
            if st.form_submit_button("Analyze customer", use_container_width=True):
                run({k: st.session_state[k] for k in DEFAULTS})

# ------------------------------------------------------------------ results
with right:
    r = st.session_state.get("result")
    if not r:
        with st.container(border=True):
            st.markdown('<div class="empty"><b>No analysis yet</b>Fill in the customer details and select '
                        'Analyze customer, or load an example to see a result.</div>', unsafe_allow_html=True)
    else:
        inputs, prob, drivers, customer = r["inputs"], r["prob"], r["drivers"], r["customer"]
        label, color = tier(prob)
        annual = inputs["MonthlyCharges"] * 12
        head, sub = {
            "High risk": ("Likely to leave", "Above the decision threshold. Start the retention plan below."),
            "Medium risk": ("Some churn risk", "Below the threshold, but proactive engagement can help."),
            "Low risk": ("Likely to stay", "Churn probability is low. Continue normal engagement."),
        }[label]

        with st.container(border=True):
            c1, c2 = st.columns([5, 6])
            with c1:
                fig = go.Figure(go.Pie(values=[prob, 1 - prob], hole=0.8, sort=False, direction="clockwise",
                                       marker=dict(colors=[color, "#e9edf2"]), textinfo="none", hoverinfo="skip"))
                fig.update_layout(height=230, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  annotations=[dict(text=f"<b>{prob:.0%}</b>", x=0.5, y=0.52, showarrow=False,
                                                    font=dict(size=40, color=INK)),
                                               dict(text="churn probability", x=0.5, y=0.36, showarrow=False,
                                                    font=dict(size=12, color=MUTED))])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(f'<div class="legend">Low below {LOW:.0%} · Medium to {THRESHOLD:.0%} · High above</div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<span class="pill" style="background:{color}1a;color:{color}">{label}</span>'
                            f'<div class="head">{head}</div><div class="sub">{sub}</div>'
                            f'<div class="stat"><span>Annual revenue at stake</span><b>${annual:,.0f}</b></div>'
                            f'<div class="stat"><span>Expected annual loss</span><b>${annual * prob:,.0f}</b></div>',
                            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="sec">What is driving this score</div>', unsafe_allow_html=True)
            top = drivers.head(6).iloc[::-1]
            bar = go.Figure(go.Bar(x=top["impact"], y=top["driver"], orientation="h",
                                   marker_color=[RED if v > 0 else GREEN for v in top["impact"]],
                                   text=[f"{v:+.2f}" for v in top["impact"]], textposition="outside",
                                   cliponaxis=False))
            bar.update_layout(height=270, margin=dict(l=0, r=40, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False),
                              yaxis=dict(tickfont=dict(color=INK, size=13)), font=dict(family="IBM Plex Sans"))
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
            st.caption("Red pushes churn risk up, green pushes it down (SHAP, log-odds).")

        with st.container(border=True):
            st.markdown('<div class="sec">Recommended actions</div>', unsafe_allow_html=True)
            if prob >= LOW:
                for t, w in playbook(inputs)[:4]:
                    st.markdown(f'<div class="action"><b>{t}</b><div>{w}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="action"><b>No action needed</b><div>Keep standard engagement.</div></div>',
                            unsafe_allow_html=True)
            report = customer.copy()
            report.insert(0, "churn_probability", round(prob, 4))
            report.insert(1, "risk_tier", label)
            report["annual_revenue_at_stake"] = annual
            st.download_button("Download report (CSV)", report.to_csv(index=False).encode("utf-8"),
                               "churnguard_report.csv", "text/csv")
