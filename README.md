# Fraud Detection AI Project

## Overview
This project demonstrates a complete **end-to-end AI pipeline** for detecting fraudulent transactions.  
It covers **synthetic data generation**, **model training**, **evaluation**, and **deployment using FastAPI**, all containerized with **Docker** for portability and scalability.

## Problem Definition
**Task:** Detect fraudulent transactions using synthetic financial data.  
**Type:** Binary Classification (`fraudulent` vs `legitimate`)  
**Goal:** Predict whether a transaction is fraudulent based on transaction and user behavioral patterns.

## Data: Synthetic Data Generation

The dataset is synthetically generated to simulate real-world financial transactions.

| Feature | Description |
|----------|-------------|
| `user_id` | Unique user identifier |
| `amount` | Transaction amount |
| `time_delta` | Time since last transaction |
| `device_trust_score` | Device reliability score |
| `location_risk_score` | Geo-based risk score |
| `merchant_category` | Vendor type |
| `transactions_last_hour` | Transaction frequency |
| `avg_transaction_amount` | User’s average spending |
| `velocity_score` | Transaction speed score |
| `is_fraud` | Target label (1 = fraud, 0 = normal) |

Synthetic generation ensures privacy and dataset balance.

---

## Backend Logic
- Data preprocessing and feature encoding
- Model: **LightGBMClassifier**
  - Excellent performance on tabular data
  - Handles categorical/numerical features efficiently
  - High interpretability and speed
- Model serialized using `joblib` → stored in `models/trained/`

---

## Evaluation Framework
Performance metrics include:

- ROC-AUC Score
- Precision / Recall / F1-score
- Confusion Matrix
- Accuracy

### Example Output:
```json
{
  "roc_auc": 0.82,
  "classification_report": {
    "0": {"precision": 0.94, "recall": 0.85, "f1-score": 0.89},
    "1": {"precision": 0.78, "recall": 0.88, "f1-score": 0.82},
    "accuracy": 0.86
  },
  "confusion_matrix": [[850, 50], [30, 70]]
}
```

---

## Deployment (FastAPI + Docker)

### Run Locally
```bash
uvicorn api.app:app --reload
```
API runs at → http://127.0.0.1:8000

### Predict Example
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "user_id": 123,
  "amount": 950.25,
  "time_delta": 2.5,
  "device_trust_score": 0.2,
  "location_risk_score": 0.8,
  "merchant_category": "electronics",
  "transactions_last_hour": 3,
  "avg_transaction_amount": 200.0,
  "velocity_score": 1.4
}'
```

**Response:**
```json
{
  "fraud_probability": 0.91,
  "is_fraud": true,
  "decision_threshold": 0.35
}
```

---

## Docker Setup

### Build Image
```bash
docker build -t fraud-detection-api .
```

### Run Container
```bash
docker run -p 8000:8000 fraud-detection-api
```

Access API → http://localhost:8000

---

## Lessons Learned
- **Data imbalance** severely affects recall → applied undersampling.
- **Feature scaling** improves convergence and consistency.
- **LightGBM** outperformed Logistic Regression and Random Forest.
- **Feature importance** improved model explainability.

---

## What I Would Do Differently
- Use **SMOTE** or **GAN-based synthetic augmentation**.
- Implement **drift detection** for real-time fraud changes.
- Integrate **MLflow** for experiment tracking.
- Deploy serverlessly via **AWS Lambda** or **Azure Container Apps**.

---

## Tech Stack
| Category | Tools |
|-----------|--------|
| **Language** | Python 3.10 |
| **Framework** | FastAPI |
| **ML Library** | LightGBM, Scikit-learn |
| **Data** | Pandas, NumPy |
| **Evaluation** | ROC-AUC, F1, Confusion Matrix |
| **Deployment** | Docker, Uvicorn |
| **Visualization** | Matplotlib, Seaborn (optional) |

---

## Steps

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python main.py
```

### Start API
```bash
uvicorn api.app:app --reload
```

---
