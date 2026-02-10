"""Streamlit demo: customer profile inputs, propensity score, ranked offers."""
import pandas as pd
import streamlit as st

from src.app.ui_components import render_offer_cards, render_propensity
from src.serving.predict import predict
from src.utils.config import get_app_config

st.set_page_config(
    page_title="Financial Offer Propensity & Ranking",
    page_icon="📊",
    layout="wide",
)

cfg = get_app_config()
app_cfg = cfg.get("app", {})
offers = cfg.get("offers", [])

st.title(app_cfg.get("title", "Financial Offer Propensity & Ranking"))
st.markdown("Simulate a customer profile and see predicted acceptance propensity and ranked offers.")

# Defaults matching UCI Bank Marketing schema
defaults = {
    "age": 40,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "balance": 1000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

with st.sidebar:
    st.header("Customer profile")
    age = st.number_input("Age", min_value=0, max_value=120, value=defaults["age"])
    job = st.selectbox(
        "Job",
        ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
        index=0,
    )
    marital = st.selectbox("Marital", ["divorced", "married", "single", "unknown"], index=1)
    education = st.selectbox(
        "Education",
        ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"],
        index=6,
    )
    balance = st.number_input("Balance", value=defaults["balance"], step=100)
    housing = st.radio("Housing loan", ["yes", "no"], index=1 if defaults["housing"] == "no" else 0)
    loan = st.radio("Personal loan", ["yes", "no"], index=1)
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"], index=0)
    duration = st.number_input("Last contact duration (sec)", min_value=0, value=defaults["duration"])
    campaign = st.number_input("Contacts this campaign", min_value=0, value=defaults["campaign"])
    pdays = st.number_input("Days since last contact (-1 = none)", value=defaults["pdays"])
    previous = st.number_input("Contacts before campaign", min_value=0, value=defaults["previous"])
    poutcome = st.selectbox("Previous outcome", ["failure", "nonexistent", "success", "unknown"], index=1)
    day = 15
    month = "may"
    default = "no"

features = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}

if st.button("Get propensity & ranking"):
    try:
        score = predict(features)
        render_propensity(score)
        # Synthetic ranking: use propensity with small offsets per offer for demo
        ranked = []
        for i, off in enumerate(offers):
            # Slight variation so ranking is visible (term_deposit = base score)
            delta = 0.02 * (i - 2) if i < len(offers) else 0
            ranked.append({
                "rank": i + 1,
                "offer_id": off.get("id", ""),
                "label": off.get("label", ""),
                "description": off.get("description", ""),
                "score": min(1.0, max(0.0, score + delta)),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        st.subheader("Ranked offers")
        render_offer_cards(ranked)
    except FileNotFoundError as e:
        st.error(f"Model not found. Train first: `make train` — {e}")
    except Exception as e:
        st.exception(e)
