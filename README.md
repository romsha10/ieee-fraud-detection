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
