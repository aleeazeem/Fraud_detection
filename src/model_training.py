import lightgbm as lgb
from joblib import dump
import os
from sklearn.pipeline import Pipeline

def train_fraud_model(preprocessor, X_train, y_train):
    os.makedirs("models/trained", exist_ok=True)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    dump(pipeline, "models/trained/fraud_model.pkl")
    print("LightGBM model trained and saved at models/trained/fraud_model.pkl")
    return pipeline
