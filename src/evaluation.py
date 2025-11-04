import json
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import os

def evaluate_model(model, X_test, y_test, threshold=0.35):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": pr_auc,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    os.makedirs("models/reports", exist_ok=True)
    with open("models/reports/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation complete — ROC-AUC: {metrics['roc_auc']:.3f} | PR-AUC: {pr_auc:.3f}")
    return metrics
