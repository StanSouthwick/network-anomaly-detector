import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import mlflow
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import logging
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Do loading data



def train_isolation_forest(X_train):
    
    # Using the full labelled dataset before the train/test split
    attack_proportion = (y != 0).sum() / len(y)
    contamination_production = float(min(attack_proportion, 0.5))  # Set contamination to the attack proportion, capped at 0.5

    # Save the calculated contamination for later use
    joblib.dump(contamination_production, '../models/iforest_contamination.pkl')
    logger.info(f"Calculated contamination for Isolation Forest: {contamination_production:.4f}")

    # Train the Isolation Forest model with the calculated contamination
    iforest_production_model = IsolationForest(n_estimators=100, contamination=contamination_production, random_state=42)
    iforest_production_model.fit(benign_traffic)
    logger.info("Trained Isolation Forest model on benign traffic.")
    joblib.dump(iforest_production_model, '../models/iforest_production_model.pkl')