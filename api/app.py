from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model pipeline
model = joblib.load("models/trained/fraud_model.pkl")

class Transaction(BaseModel):
    user_id: int
    amount: float
    time_delta: float
    device_trust_score: float
    location_risk_score: float
    merchant_category: str
    transactions_last_hour: int
    avg_transaction_amount: float
    velocity_score: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is live"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    proba = model.predict_proba(data)[0, 1]
    prediction = int(proba >= 0.35)
    return {
        "fraud_probability": round(float(proba), 3),
        "is_fraud": bool(prediction),
        "decision_threshold": 0.35
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
