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

This project builds a complete, production-grade **financial fraud detection pipeline** on 590,000 real banking transactions from the IEEE-CIS Fraud Detection dataset.

It covers the entire lifecycle of a fraud detection system — raw data ingestion, exploratory analysis, feature engineering, class imbalance handling, model training, explainability, real-time scoring simulation, a live analyst dashboard, and a SQL-backed production pipeline.

---

## Why This Problem Matters

> 💡 **Impact:** Financial fraud costs the global banking industry over **$40 billion annually.**

Every percentage point improvement in fraud recall translates directly into millions of dollars saved. But there is a tradeoff — flagging too many legitimate transactions as fraud destroys customer trust.

The goal is a model that meets four critical requirements:
* **High Recall:** Catches as much fraud as possible.
* **High Precision:** Does not harass legitimate customers.
* **SHAP Explainability:** Can explain every decision it makes.
* **Real-Time Performance:** Can score a transaction in milliseconds.

This project addresses all four requirements.

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)

| File | Rows | Columns | Description |
| :--- | :--- | :--- | :--- |
| `train_transaction.csv` | 590,540 | 394 | Core transaction data |
| `train_identity.csv` | 144,233 | 41 | Device and identity data |

### Class Distribution
* **Legitimate transactions:** ~96.5%
* **Fraudulent transactions:** ~3.5%

---

## Project Architecture

1.  **Raw Data (CSV)**
2.  **Data Loading & Merging**
3.  **Exploratory Data Analysis**
4.  **Feature Engineering**
5.  **Preprocessing** *(Encoding + Missing Value Handling)*
6.  **Train/Test Split** *(80/20, stratified)*
7.  **SMOTE** *(applied to training set only)*
8.  **Model Training** *(XGBoost / Random Forest / Isolation Forest)*
9.  **Evaluation** *(Precision / Recall / F1 / AUC-ROC)*
10. **SHAP Explainability**
11. **Real-Time Scoring Pipeline**
12. **SQL Database Integration**
13. **Streamlit Fraud Analyst Dashboard** $\rightarrow$ *Deployed on Hugging Face*

---

## Features Engineered

### 🕒 Time-Based Features
| Feature | Description | Why It Matters |
| :--- | :--- | :--- |
| `hour` | Hour of day (0–23) | Fraud spikes at unusual hours |
| `day_of_week` | Day of week (0–6) | Weekend patterns differ |
| `week` | Week number | Seasonal patterns |

### 💵 Amount-Based Features
| Feature | Description | Why It Matters |
| :--- | :--- | :--- |
| `TransactionAmt_log` | Log-transformed amount | Reduces skewness |
| `TransactionAmt_decimal` | Decimal portion of amount | Fraudsters use .00 or .99 |
| `is_round_amount` | Is amount a round number? | Round amounts are a fraud signal |

### ⚡ Velocity Features
| Feature | Description | Why It Matters |
| :--- | :--- | :--- |
| `card1_count` | Total transactions on this card | High velocity = suspicious |
| `card1_amt_mean` | Average spend on this card | Baseline normal behaviour |
| `amt_deviation` | Deviation from card average | Key fraud signal |

### 📱 Device Fingerprinting Proxies
| Feature | Description | Why It Matters |
| :--- | :--- | :--- |
| `DeviceType_count` | Frequency of this device type | Rare devices are suspicious |
| `email_unique_cards` | Unique cards per email domain | Proxy for synthetic identity fraud |

---

## Models Built

### 1. XGBoost (Primary Model)
* Industry standard for fraud detection
* Utilizes `scale_pos_weight` for class imbalance
* Under 50ms inference for real-time scoring

### 2. Random Forest
* Ensemble of independent decision trees
* Utilizes `class_weight='balanced'` for imbalance handling
* Robust to overfitting

### 3. Isolation Forest
* Unsupervised anomaly detection
* Requires no fraud labels
* Detects fraud by isolating outliers

---

## Handling Class Imbalance

Two-pronged approach:

### SMOTE — Synthetic Minority Oversampling
Creates synthetic fraud examples by interpolating between existing fraud cases. Applied only to training data — never test data.

```python
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
