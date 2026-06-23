# 🏦 IEEE-CIS Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

🚀 **Live Demo:** [Click here to open the dashboard](https://huggingface.co/spaces/YOUR_USERNAME/fraud-detection)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Why This Problem Matters](#why-this-problem-matters)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Features Engineered](#features-engineered)
- [Models Built](#models-built)
- [Handling Class Imbalance](#handling-class-imbalance)
- [SHAP Explainability](#shap-explainability)
- [Real-Time Scoring](#real-time-scoring)
- [Streamlit Dashboard](#streamlit-dashboard)
- [SQL Integration](#sql-integration)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This project builds a complete, production-grade **financial fraud detection pipeline**
on 590,000 real banking transactions from the IEEE-CIS Fraud Detection dataset.

It covers the entire lifecycle of a fraud detection system — raw data ingestion,
exploratory analysis, feature engineering, class imbalance handling, model training,
explainability, real-time scoring simulation, a live analyst dashboard,
and a SQL-backed production pipeline.

---

## Why This Problem Matters

> Financial fraud costs the global banking industry over **$40 billion annually.**

Every percentage point improvement in fraud recall translates directly into
millions of dollars saved. But there is a tradeoff — flagging too many legitimate
transactions as fraud destroys customer trust.

The goal is a model that:
- Catches as much fraud as possible — high **Recall**
- Does not harass legitimate customers — high **Precision**
- Can explain every decision it makes — **SHAP Explainability**
- Can score a transaction in milliseconds — **Real-Time Performance**

This project addresses all four requirements.

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train_transaction.csv` | 590,540 | 394 | Core transaction data |
| `train_identity.csv` | 144,233 | 41 | Device and identity data |

**Class distribution:**
- Legitimate transactions: ~96.5%
- Fraudulent transactions: ~3.5%

---

## Project Architecture

Raw Data (CSV)

│

▼

Data Loading & Merging

│

▼

Exploratory Data Analysis

│

▼

Feature Engineering

│

▼

Preprocessing (Encoding + Missing Value Handling)

│

▼

Train/Test Split (80/20, stratified)

│

▼

SMOTE (applied to training set only)

│

▼

Model Training (XGBoost / Random Forest / Isolation Forest)

│

▼

Evaluation (Precision / Recall / F1 / AUC-ROC)

│

▼

SHAP Explainability

│

▼

Real-Time Scoring Pipeline

│

▼

SQL Database Integration

│

▼

Streamlit Fraud Analyst Dashboard → Deployed on Hugging Face

---

## Features Engineered

### Time-Based Features
| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `hour` | Hour of day (0–23) | Fraud spikes at unusual hours |
| `day_of_week` | Day of week (0–6) | Weekend patterns differ |
| `week` | Week number | Seasonal patterns |

### Amount-Based Features
| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `TransactionAmt_log` | Log-transformed amount | Reduces skewness |
| `TransactionAmt_decimal` | Decimal portion of amount | Fraudsters use .00 or .99 |
| `is_round_amount` | Is amount a round number? | Round amounts are a fraud signal |

### Velocity Features
| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `card1_count` | Total transactions on this card | High velocity = suspicious |
| `card1_amt_mean` | Average spend on this card | Baseline normal behaviour |
| `amt_deviation` | Deviation from card average | Key fraud signal |

### Device Fingerprinting Proxies
| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `DeviceType_count` | Frequency of this device type | Rare devices are suspicious |
| `email_unique_cards` | Unique cards per email domain | Proxy for synthetic identity fraud |

---

## Models Built

### 1. XGBoost (Primary Model)
- Industry standard for fraud detection
- `scale_pos_weight` for class imbalance
- Under 50ms inference for real-time scoring

### 2. Random Forest
- Ensemble of independent decision trees
- `class_weight='balanced'` for imbalance handling
- Robust to overfitting

### 3. Isolation Forest
- Unsupervised anomaly detection
- Requires no fraud labels
- Detects fraud by isolating outliers

---

## Handling Class Imbalance

Two-pronged approach:

**SMOTE — Synthetic Minority Oversampling**
Creates synthetic fraud examples by interpolating between existing fraud cases.
Applied only to training data — never test data.

```python
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

**Weighted Loss Functions**
XGBoost `scale_pos_weight` penalises missing fraud cases more heavily.

```python
scale = (y_train == 0).sum() / (y_train == 1).sum()
```

---

## SHAP Explainability

Every flagged transaction gets a plain-English explanation:
🚨 FRAUD — Score: 87.3%
🔴 Top reasons flagged:

→ amt_deviation: +0.4821      (spent far above card average)

→ hour: +0.3102               (transaction at 3am)

→ is_round_amount: +0.2841    (exactly $500.00)
🟢 Top reasons against fraud:

→ TransactionAmt_log: -0.0821

→ day_of_week: -0.0412

---

## Real-Time Scoring

Every transaction scored in under 50ms:

```python
result = score_transaction(raw_transaction, model, explainer, feature_cols)
# Returns:
# {
#   'fraud_probability': 0.873,
#   'decision': 'FRAUD',
#   'risk_level': 'HIGH',
#   'top_reasons': {...},
#   'latency_ms': 23.4
# }
```

---

## Streamlit Dashboard

Four panels:

| Panel | What it shows |
|-------|--------------|
| 📊 Overview | KPIs, fraud score distribution, risk breakdown, hourly patterns |
| 🚨 Fraud Alerts | Filterable investigation queue with SHAP explanations |
| 🔍 Transaction Scorer | Live scoring for any transaction |
| 📈 Model Performance | All three models compared side by side |

---

## SQL Integration

Three production tables:

| Table | Purpose |
|-------|---------|
| `transactions` | Every scored transaction with fraud probability |
| `fraud_alerts` | Investigation queue with status tracking |
| `model_performance` | Daily model metrics for drift monitoring |

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualisations |
| Scikit-learn | Preprocessing, RF, Isolation Forest, metrics |
| XGBoost | Primary fraud detection model |
| Imbalanced-learn | SMOTE implementation |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |
| SQLite | Database simulation |
| Hugging Face Spaces | Dashboard deployment |
| Google Colab | Development environment |

---

## Project Structure
ieee-fraud-detection/

│

├── fraud_detection.ipynb   ← Full pipeline notebook

├── app.py                  ← Streamlit dashboard

├── requirements.txt        ← Python dependencies

├── .gitignore              ← Excludes data and model files

└── README.md               ← This file

---

## How to Run

### Option 1 — Live Dashboard
👉 [Open on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/fraud-detection)

### Option 2 — Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/ieee-fraud-detection
cd ieee-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```

Note: You need to add the model files manually since they
are not tracked by Git. Download the dataset from Kaggle
and run the full notebook first.

### Option 3 — Run the Notebook
Open `fraud_detection.ipynb` in Google Colab and run all cells top to bottom.

---

## Results

| Model | Precision | Recall | F1 Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| XGBoost | — | — | — | — |
| Random Forest | — | — | — | — |
| Isolation Forest | — | — | — | — |

> Fill in your actual numbers after running the notebook.

---

## Key Learnings

1. **Accuracy is a lie with imbalanced data** — always use Precision, Recall, F1, AUC-ROC
2. **SMOTE only on training data** — applying it to test data inflates metrics artificially
3. **Feature engineering beats hyperparameter tuning** — velocity features mattered more than model choice
4. **Explainability is not optional** — regulated industries require reasons for every decision
5. **Real-time scoring needs a pipeline** — preprocessing and inference must be packaged together

---

## Future Improvements

- [ ] FastAPI REST endpoint for production scoring
- [ ] Neural network comparison (PyTorch)
- [ ] Model drift detection and alerting
- [ ] Graph-based features for fraud ring detection
- [ ] Analyst feedback loop for model retraining
- [ ] AWS / GCP deployment with MySQL

---

## Author

**Your Name**
- 🔗 LinkedIn: [your-linkedin-url]
- 💻 GitHub: [your-github-url]
- 📧 Email: [your-email]

---

## License

MIT License

Dataset provided by Vesta Corporation via Kaggle.
Subject to Kaggle competition rules — do not redistribute raw data files.
