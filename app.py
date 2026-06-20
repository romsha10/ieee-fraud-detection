import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import json
import os

# Page Config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏦",
    layout="wide"
)

# Base Directory Fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model and Data
@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

@st.cache_data
def load_data():
    X = pd.read_csv(os.path.join(BASE_DIR, "sample_transactions.csv"))
    y = pd.read_csv(os.path.join(BASE_DIR, "sample_labels.csv"))
    with open(os.path.join(BASE_DIR, "model_results.json"), "r") as f:
        results = json.load(f)
    return X, y, results

model, feature_cols               = load_model()
X_sample, y_sample, model_results = load_data()

# Compute Scores
probs       = model.predict_proba(X_sample[feature_cols])[:, 1]
decisions   = ["FRAUD" if p >= 0.5 else "LEGITIMATE" for p in probs]
risk_levels = ["HIGH" if p > 0.7 else "MEDIUM" if p > 0.4 else "LOW" for p in probs]

results_df = X_sample.copy()
results_df["fraud_probability"] = probs
results_df["decision"]          = decisions
results_df["risk_level"]        = risk_levels
results_df["actual"]            = y_sample.values.flatten()

xgb_results = model_results["xgb"]
rf_results  = model_results["rf"]
iso_results = model_results["iso"]

# Sidebar
st.sidebar.title("Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["How to Use", "Overview", "Fraud Alerts", "Transaction Scorer", "Model Performance"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Transactions:** {len(results_df):,}")
st.sidebar.markdown(f"**Fraud Flagged:** {(results_df['decision']=='FRAUD').sum():,}")
st.sidebar.markdown(f"**Fraud Rate:** {results_df['actual'].mean()*100:.2f}%")

# =======================================================
# PAGE 0 - HOW TO USE
# =======================================================
if page == "How to Use":
    st.title("How to Use This Dashboard")
    st.markdown("---")

    st.markdown("""
    ## What Is This?

    This is a **fraud detection system** for banking transactions. It uses a
    machine learning model (XGBoost) trained on 590,000 real transactions to
    predict whether a new transaction is **fraudulent** or **legitimate**,
    and explains **why** it made that decision.

    Think of it as a tool a bank's fraud analyst would use every day to
    review suspicious transactions and decide what to do about them.
    """)

    st.markdown("---")
    st.markdown("## What Each Page Does")

    st.markdown("""
    ### Overview
    **What it shows:** A bird's-eye view of all transactions in the sample -
    how many are fraud, how many are legitimate, and patterns like which
    hours of the day see more fraud.

    **Why it matters:** This is your starting point. It tells you the
    overall health of the system - is fraud spiking? Is it concentrated
    at certain times?

    **How to read it:**
    - The **KPI cards** at the top give quick numbers (total transactions, fraud flagged, etc.)
    - The **Fraud Score Distribution** chart shows how confident the model is -
      red bars on the right mean the model is very sure something is fraud
    - The **Risk Level** chart groups transactions into HIGH / MEDIUM / LOW risk
    - The **Fraud Rate by Hour** chart shows if fraud clusters at certain times of day

    ---

    ### Fraud Alerts
    **What it shows:** A list of transactions the model has flagged as fraud,
    sorted by how risky they are.

    **Why it matters:** This is the **investigation queue** - exactly what
    a fraud analyst opens every morning to decide which cases to investigate first.

    **How to use it:**
    1. Use the **Risk Level filter** to focus on HIGH risk cases first
    2. Use the **score slider** to only see cases above a certain confidence level
    3. Pick a row number in "Investigate a Specific Alert"
    4. Click "Run SHAP Explanation" - this shows you exactly why the
       model thinks this transaction is fraud (e.g. amount is far above what
       this card normally spends, or the transaction happened at 3am)
    5. Based on the explanation click "Confirm as Fraud" if it looks
       genuinely suspicious, or "Mark as False Positive" if it looks
       like a normal transaction the model got wrong

    **The goal:** Help a human analyst make a fast, informed decision instead
    of staring at raw numbers.

    ---

    ### Transaction Scorer
    **What it shows:** A live "what if" tool - enter the details of any
    transaction and instantly see if the model would flag it as fraud.

    **Why it matters:** This simulates what happens in real time when a
    new transaction arrives - the model scores it in milliseconds.

    **How to use it:**
    1. Enter a **Transaction Amount** (try something unusual like $9999)
    2. Set the **Hour of Day** (try 3 AM vs 2 PM and compare the results)
    3. Set **Day of Week**, **Card Transaction Count**, and **Amount Deviation**
    4. Tick "Is Round Amount?" if testing a round number like $500.00
    5. Click "Score Transaction Now"
    6. Read the result - Decision (Fraud/Legitimate), Fraud Score %, and Risk Level

    **Try this experiment:**
    Set amount to $50, hour to 2 PM, deviation to 0 - likely LEGITIMATE.
    Now change amount to $5000, hour to 3 AM, deviation to 5 - likely FRAUD.
    This shows you which factors the model cares about most.

    ---

    ### Model Performance
    **What it shows:** How well each of the three models (XGBoost, Random
    Forest, Isolation Forest) performs using standard evaluation metrics.

    **Why it matters:** Not all models are equal. This page lets you compare
    them and understand why XGBoost was chosen as the primary model.

    **How to read the metrics:**
    - **Precision** - Of all transactions flagged as fraud, what % were actually fraud?
      High precision means few false alarms on legitimate customers.
    - **Recall** - Of all actual fraud cases, what % did the model catch?
      High recall means fewer frauds slipping through undetected.
    - **F1 Score** - A single number balancing both precision and recall.
    - **AUC-ROC** - Overall ability to distinguish fraud from legitimate.
      Closer to 1.0 is better. 0.5 means the model is no better than random guessing.

    **Why both precision AND recall matter:**
    A model that flags everything as fraud has perfect recall but terrible precision
    - it would block every legitimate customer. A model that flags nothing has
    perfect precision but catches zero fraud. The best model balances both.
    """)

    st.markdown("---")
    st.info("""
    Quick Start - New here?
    Start with Overview to get the big picture,
    then go to Fraud Alerts to see the investigation workflow in action,
    and try Transaction Scorer to experiment with your own transaction values.
    """)

# =======================================================
# PAGE 1 - OVERVIEW
# =======================================================
elif page == "Overview":
    st.title("Fraud Detection - Overview Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(results_df):,}")
    col2.metric("Fraud Flagged",      f"{(results_df['decision']=='FRAUD').sum():,}")
    col3.metric("High Risk Alerts",   f"{(results_df['risk_level']=='HIGH').sum():,}")
    col4.metric("Avg Fraud Score",    f"{results_df['fraud_probability'].mean()*100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fraud Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(results_df[results_df["actual"]==0]["fraud_probability"],
                bins=30, alpha=0.7, color="#2ecc71", label="Legitimate")
        ax.hist(results_df[results_df["actual"]==1]["fraud_probability"],
                bins=30, alpha=0.7, color="#e74c3c", label="Fraud")
        ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold")
        ax.set_xlabel("Fraud Probability")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Risk Level Breakdown")
        risk_counts = results_df["risk_level"].value_counts()
        colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(risk_counts.index, risk_counts.values,
                color=[colors.get(r, "#3498db") for r in risk_counts.index])
        ax2.set_ylabel("Count")
        for i, v in enumerate(risk_counts.values):
            ax2.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.subheader("Fraud Rate by Hour of Day")
    if "hour" in results_df.columns:
        hour_fraud = results_df.groupby(results_df["hour"].astype(int))["actual"].mean() * 100
        fig3, ax3  = plt.subplots(figsize=(12, 4))
        ax3.bar(hour_fraud.index, hour_fraud.values, color="#3498db")
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("Fraud Rate %")
        st.pyplot(fig3)
        plt.close()

# =======================================================
# PAGE 2 - FRAUD ALERTS
# =======================================================
elif page == "Fraud Alerts":
    st.title("Fraud Alerts - Investigation Queue")
    st.markdown("---")

    fraud_df = results_df[results_df["decision"] == "FRAUD"].copy()
    fraud_df = fraud_df.sort_values("fraud_probability", ascending=False)
    fraud_df["fraud_score_%"] = (fraud_df["fraud_probability"] * 100).round(1)

    col1, col2 = st.columns(2)
    risk_filter     = col1.multiselect("Filter by Risk Level",
                                        ["HIGH", "MEDIUM", "LOW"],
                                        default=["HIGH", "MEDIUM"])
    score_threshold = col2.slider("Minimum Fraud Score %", 50, 100, 50)

    filtered = fraud_df[
        (fraud_df["risk_level"].isin(risk_filter)) &
        (fraud_df["fraud_score_%"] >= score_threshold)
    ]

    st.markdown(f"**Showing {len(filtered)} alerts**")
    display_cols = ["fraud_score_%", "risk_level", "TransactionAmt",
                    "hour", "amt_deviation", "is_round_amount"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].reset_index(drop=True),
                 use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Investigate a Specific Alert")
    alert_idx = st.number_input("Enter row number to investigate",
                                 min_value=0,
                                 max_value=max(len(filtered)-1, 0),
                                 value=0)

    if st.button("Run SHAP Explanation"):
        selected = filtered.iloc[int(alert_idx)]
        with st.spinner("Computing SHAP explanation..."):
            explainer_dash = shap.TreeExplainer(model)
            txn_df    = pd.DataFrame([selected[feature_cols]])
            shap_vals = explainer_dash.shap_values(txn_df)
            feat_shap = pd.Series(shap_vals[0], index=feature_cols)
            top_fraud = feat_shap.nlargest(5)
            top_legit = feat_shap.nsmallest(5)

            col_x, col_y = st.columns(2)
            with col_x:
                st.markdown("**Top reasons flagged as FRAUD:**")
                for feat, val in top_fraud.items():
                    st.markdown(f"- `{feat}`: impact `{val:+.4f}`")
            with col_y:
                st.markdown("**Top reasons against fraud:**")
                for feat, val in top_legit.items():
                    st.markdown(f"- `{feat}`: impact `{val:+.4f}`")

    st.markdown("---")
    col_p, col_q = st.columns(2)
    if col_p.button("Mark as False Positive"):
        st.success("Marked as False Positive - removed from queue")
    if col_q.button("Confirm as Fraud"):
        st.error("Confirmed as Fraud - case escalated to investigation team")

# =======================================================
# PAGE 3 - TRANSACTION SCORER
# =======================================================
elif page == "Transaction Scorer":
    st.title("Real-Time Transaction Scorer")
    st.markdown("Score any transaction instantly using the live model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    amt       = col1.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    hour_val  = col2.slider("Hour of Day", 0, 23, 14)
    day_val   = col3.slider("Day of Week", 0, 6, 3)

    col4, col5 = st.columns(2)
    card_count = col4.number_input("Card Transaction Count", min_value=1, value=10)
    amt_dev    = col5.number_input("Amount Deviation", value=0.5)
    round_amt  = st.checkbox("Is Round Amount?")

    if st.button("Score Transaction Now"):
        raw = {col: 0 for col in feature_cols}
        raw.update({
            "TransactionAmt":         amt,
            "TransactionAmt_log":     np.log1p(amt),
            "TransactionAmt_decimal": amt - int(amt),
            "is_round_amount":        int(round_amt),
            "hour":                   hour_val,
            "day_of_week":            day_val,
            "card1_count":            card_count,
            "amt_deviation":          amt_dev,
        })
        txn_df = pd.DataFrame([raw])[feature_cols]
        prob   = model.predict_proba(txn_df)[0][1]
        risk   = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        dec    = "FRAUD" if prob >= 0.5 else "LEGITIMATE"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Decision",    dec)
        r2.metric("Fraud Score", f"{prob*100:.1f}%")
        r3.metric("Risk Level",  risk)
        st.progress(float(prob))

# =======================================================
# PAGE 4 - MODEL PERFORMANCE
# =======================================================
elif page == "Model Performance":
    st.title("Model Performance Metrics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("XGBoost Precision", f"{xgb_results['precision']:.4f}")
    col2.metric("XGBoost Recall",    f"{xgb_results['recall']:.4f}")
    col3.metric("XGBoost F1",        f"{xgb_results['f1']:.4f}")
    col4.metric("XGBoost AUC-ROC",   f"{xgb_results['auc']:.4f}")

    st.markdown("---")
    st.subheader("Model Comparison")
    compare_df = pd.DataFrame([xgb_results, rf_results, iso_results])
    compare_df = compare_df.set_index("model").round(4)
    st.dataframe(compare_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    compare_df[["precision", "recall", "f1"]].plot(
        kind="bar", ax=ax,
        color=["#3498db", "#e74c3c", "#2ecc71"],
        edgecolor="black"
    )
    ax.set_title("Model Comparison - Precision, Recall, F1")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
