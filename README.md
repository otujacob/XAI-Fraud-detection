# XAI Fraud Detection — Nigerian Inter-Bank Payments

**COM 752 MSc Dissertation**  
Otu Samuel Jacob | s25007038 | Wrexham University | 2025/2026

## Overview

This Streamlit application demonstrates the Explainable AI framework with Human-in-the-Loop feedback for real-time fraud detection in Nigerian inter-bank payment systems.

## Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Dataset summary, framework architecture, key results |
| 📊 Model Results | Full performance table and charts (Table 4.1) |
| 🔍 SHAP Explanations | Three real test-set transactions with SHAP attribution |
| 🔄 HITL Feedback | Human-in-the-Loop simulation results across 3 cycles |
| ⚡ Live Scoring | Upload new CSV or run demo dataset through the full pipeline |

## Local Setup

```bash
git clone <your-repo-url>
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Click Deploy

## Dataset

The full NIBSS-calibrated dataset (293MB) is not included in this repository.  
A demo sample (528 transactions) is included in `data/demo_transactions.csv`.  
To score the full dataset, use the **Upload CSV** tab in Live Scoring.

## Reference

Idowu, G.A. & Owolabi, J.E. (2026). Ensemble-Based Fraud Detection in Nigerian Banking.  
*Unilag Journal of Mathematics and Applications*, 6(1), 107–129.
