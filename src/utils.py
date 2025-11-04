import os

def ensure_directories():
    for folder in ["data/synthetic", "models/trained", "models/reports"]:
        os.makedirs(folder, exist_ok=True)
