
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

# Load Model and Data
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

@st.cache_data
def load_data():
    X = pd.read_csv("sample_transactions.csv")
    y = pd.read_csv("sample_labels.csv")
    with open("model_results.json", "r") as f:
        results = json.load(f)
    return X, y, results

model, feature_cols          = load_model()
X_sample, y_sample, model_results = load_data()

# Compute scores
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
    ["Overview", "Fraud Alerts", "Transaction Scorer", "Model Performance"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Transactions:** {len(results_df):,}")
st.sidebar.markdown(f"**Fraud Flagged:** {(results_df['decision']=='FRAUD').sum():,}")
st.sidebar.markdown(f"**Fraud Rate:** {results_df['actual'].mean()*100:.2f}%")

# PAGE 1 - OVERVIEW
if page == "Overview":
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

# PAGE 2 - FRAUD ALERTS
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
        selected  = filtered.iloc[int(alert_idx)]
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

# PAGE 3 - TRANSACTION SCORER
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

    if st.button("⚡ Score Transaction Now"):
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

# PAGE 4 - MODEL PERFORMANCE
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
