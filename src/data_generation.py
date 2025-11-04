import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.05, random_state=42):
    np.random.seed(random_state)
    os.makedirs("data/synthetic", exist_ok=True)

    df = pd.DataFrame({
        "transaction_id": range(1, n_samples + 1),
        "user_id": np.random.randint(1, 1000, n_samples),
        "amount": np.random.exponential(scale=200, size=n_samples),
        "time_delta": np.random.exponential(scale=50, size=n_samples),
        "device_trust_score": np.random.uniform(0, 1, n_samples),
        "location_risk_score": np.random.uniform(0, 1, n_samples),
        "merchant_category": np.random.choice(["electronics", "fashion", "grocery", "travel"], n_samples)
    })

    df["transactions_last_hour"] = np.random.poisson(lam=2, size=n_samples)
    df["avg_transaction_amount"] = np.random.exponential(scale=150, size=n_samples)
    df["velocity_score"] = np.log1p(df["transactions_last_hour"]) * df["location_risk_score"]

    fraud_prob = (
        0.3 * (df["amount"] > 500).astype(int) +
        0.2 * (df["time_delta"] < 10).astype(int) +
        0.3 * (df["location_risk_score"] > 0.7).astype(int) +
        0.2 * (df["device_trust_score"] < 0.3).astype(int)
    )

    df["fraud_probability"] = fraud_prob / fraud_prob.max()
    df["is_fraud"] = (df["fraud_probability"] > np.quantile(df["fraud_probability"], 1 - fraud_ratio)).astype(int)

    df.to_csv("data/synthetic/synthetic_data.csv", index=False)
    print("Synthetic data generated at data/synthetic/synthetic_data.csv")
    return df
