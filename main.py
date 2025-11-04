import sys
import os

from src.data_generation import generate_synthetic_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_fraud_model
from src.evaluation import evaluate_model
from src.utils import ensure_directories

def main():
    print("Starting Fraud Detection Pipeline...")
    ensure_directories()

    print("Generating synthetic data...")
    generate_synthetic_data()

    print("Preprocessing data...")
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data()

    print("Training model...")
    model = train_fraud_model(preprocessor, X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, threshold=0.35)

    print("Pipeline complete! Check models/reports/model_metrics.json for results.")

if __name__ == "__main__":
    main()
