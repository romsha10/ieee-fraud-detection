# 🏦 IEEE-CIS Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square\&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square\&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-IEEE--CIS-blue?style=flat-square\&logo=kaggle)

## 🚀 Live Demo

👉 **[Open Streamlit Dashboard](#)**

---

## 📖 Overview

A production-style fraud detection system built on the IEEE-CIS Fraud Detection dataset containing **590,540 banking transactions**.

The project covers the complete machine learning lifecycle:

* Data ingestion and merging
* Exploratory data analysis (EDA)
* Feature engineering
* Class imbalance handling
* Model development and evaluation
* Explainable AI (SHAP)
* Real-time fraud scoring
* SQL integration
* Interactive Streamlit dashboard

---

## 🎯 Business Problem

Financial fraud costs the banking industry billions annually.

An effective fraud detection system must:

* ✅ Maximise fraud detection (**Recall**)
* ✅ Minimise false alarms (**Precision**)
* ✅ Explain every decision (**SHAP**)
* ✅ Score transactions in milliseconds (**Real-Time Inference**)

This project addresses all four requirements.

---

## 📊 Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle)

| Dataset               | Rows    | Columns |
| --------------------- | ------- | ------- |
| train_transaction.csv | 590,540 | 394     |
| train_identity.csv    | 144,233 | 41      |
| Merged Dataset        | 590,540 | 434     |

### Class Distribution

| Class      | Count   | Percentage |
| ---------- | ------- | ---------- |
| Legitimate | 569,877 | 96.5%      |
| Fraud      | 20,663  | 3.5%       |

The severe class imbalance makes fraud detection a challenging machine learning problem.

---

## 🏗️ System Architecture

```text
Raw Data
    ↓
Data Processing & Merging
    ↓
EDA
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Train/Test Split
    ↓
SMOTE
    ↓
Model Training
    ↓
Model Evaluation
    ↓
SHAP Explainability
    ↓
Real-Time Scoring
    ↓
SQL Storage
    ↓
Streamlit Dashboard
```

---

## ⚙️ Feature Engineering

### Time Features

* Hour of day
* Day of week
* Week number

### Amount Features

* Log transaction amount
* Decimal amount extraction
* Round-number detection

### Velocity Features

* Card transaction counts
* Spending averages
* Spending deviations

### Device Fingerprinting

* Device frequency counts
* Device information rarity
* Email-card relationship features

**Final Feature Count:** 231

---

## 🤖 Models

### 1. XGBoost (Primary Model)

* 300 estimators
* Max depth = 6
* Learning rate = 0.05
* `scale_pos_weight`

### 2. Random Forest

* 200 trees
* Balanced class weights
* Interpretable ensemble model

### 3. Isolation Forest

* Unsupervised anomaly detection
* No labels required

---

## ⚖️ Handling Class Imbalance

### SMOTE Oversampling

```python
smote = SMOTE(
    random_state=42,
    sampling_strategy=0.3
)
```

| Stage        | Fraud   | Legitimate |
| ------------ | ------- | ---------- |
| Before SMOTE | 16,530  | 455,902    |
| After SMOTE  | 136,770 | 455,902    |

### Cost-Sensitive Learning

```python
scale_pos_weight = negative_class / positive_class
```

Used within XGBoost to increase penalties for missed fraud cases.

---

## 📈 Results

### Model Performance

| Model            | Precision | Recall | F1    | AUC-ROC |
| ---------------- | --------- | ------ | ----- | ------- |
| XGBoost          | 0.636     | 0.552  | 0.591 | 0.923   |
| Random Forest    | 0.346     | 0.561  | 0.428 | 0.877   |
| Isolation Forest | 0.165     | 0.089  | 0.116 | —       |

### Best Model: XGBoost

* Precision: 63.6%
* Recall: 55.2%
* F1 Score: 0.5909
* AUC-ROC: 0.9226

---

## 🔍 Explainable AI with SHAP

Each fraud prediction includes:

* Global feature importance
* Local transaction explanations
* Feature impact visualisations

Example:

```text
Fraud Score: 94.9%

Top Drivers:
+ C1
+ C4
+ card6

Risk Reducers:
- TransactionAmt
- C14
```

---

## ⚡ Real-Time Fraud Scoring

Average inference latency:

| Metric  | Value    |
| ------- | -------- |
| Average | 51.73 ms |
| P95     | 96.44 ms |

Example output:

```python
{
    "fraud_probability": 0.9489,
    "decision": "FRAUD",
    "risk_level": "HIGH",
    "latency_ms": 38.65
}
```

---

## 🖥️ Streamlit Dashboard

### Dashboard Modules

| Module             | Purpose               |
| ------------------ | --------------------- |
| Overview           | KPIs and fraud trends |
| Fraud Alerts       | Investigation queue   |
| Transaction Scorer | Live scoring          |
| Model Performance  | Model comparison      |

### Features

* Risk filtering
* SHAP explanations
* Alert investigation workflow
* Threshold adjustment
* Drift monitoring

---

## 🗄️ SQL Integration

Three production-style tables:

* `transactions`
* `fraud_alerts`
* `model_performance`

Supports:

* Fraud monitoring
* Investigation workflows
* Performance tracking
* Audit trails

---

## 🛠 Tech Stack

| Category           | Technology          |
| ------------------ | ------------------- |
| Language           | Python              |
| Data Processing    | Pandas, NumPy       |
| Visualisation      | Matplotlib, Seaborn |
| ML                 | Scikit-Learn        |
| Gradient Boosting  | XGBoost             |
| Imbalance Handling | SMOTE               |
| Explainability     | SHAP                |
| Dashboard          | Streamlit           |
| Database           | SQLite              |
| Deployment         | Hugging Face Spaces |

---

## 📁 Project Structure

```text
ieee-fraud-detection/
│
├── fraud_detection.ipynb
├── app.py
├── requirements.txt
├── fraud_detection.db
├── README.md
│
├── models/
│   ├── xgb_model.pkl
│   └── feature_cols.pkl
│
├── hf_deploy/
│   └── app.py
│
└── sample_data/
    ├── sample_transactions.csv
    └── sample_labels.csv
```

---

## 🚀 Running the Project

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ieee-fraud-detection
cd ieee-fraud-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run hf_deploy/app.py
```

---

## 💡 Key Learnings

* Accuracy is misleading on imbalanced datasets
* SMOTE should only be applied to training data
* Feature engineering outperformed hyperparameter tuning
* Explainability is critical in regulated industries
* Production systems require more than just ML models

---

## 🔮 Future Improvements

* FastAPI deployment
* Kafka streaming integration
* Graph-based fraud detection
* Neural network benchmarks
* Automated model drift detection
* Cloud deployment (AWS/GCP)
* Analyst feedback retraining loop

---

## 👤 Author

**Your Name**

* LinkedIn: your-linkedin-url
* GitHub: your-github-url
* Email: [your-email@example.com](mailto:your-email@example.com)

---

## 📜 License

MIT License

---

### Dataset Credit

IEEE-CIS Fraud Detection Dataset provided by Vesta Corporation through Kaggle.

Raw competition data should not be redistributed.
