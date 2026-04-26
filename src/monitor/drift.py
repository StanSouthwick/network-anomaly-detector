# Drift monitoring for network anomaly detector
import logging
import numpy as np
import mlflow
from pathlib import Path
from scipy.stats import ks_2samp
import json



logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent.parent
REFERENCE_X = BASE_DIR / "data" / "processed" / "X_train.npy"
REFERENCE_Y = BASE_DIR / "data" / "processed" / "y_train.npy"



# Function to load reference data for drift monitoring, which is the training data used to build the models
def load_reference_data():
    try:
        X_ref = np.load(REFERENCE_X)
        y_ref = np.load(REFERENCE_Y)
        logger.info("Reference data loaded successfully")
        with open(BASE_DIR / "artifacts" / "feature_names.json") as f:
            FEATURE_NAMES = json.load(f)
        return X_ref, y_ref, FEATURE_NAMES
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        raise e

# Function to compute drift metrics using Kolmogorov-Smirnov test for each feature
def compute_drift_metrics(X_ref, X_new, feature_names):

    
    drift_metrics = []
    for feature_idx, feature in enumerate(feature_names):
        statistic, p_value = ks_2samp(X_ref[:, feature_idx], X_new[:, feature_idx])
        drift_metrics.append({
            "feature": feature,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": p_value < 0.05
        })
    return drift_metrics

def log_drift_report(drift_metrics):
    mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
    mlflow.set_experiment("Network Anomaly Detector Drift Monitoring")
    total_drifted_features = sum(metric["drift_detected"] for metric in drift_metrics)
    with mlflow.start_run(run_name="Drift Report"):
        mlflow.log_param("n features", len(drift_metrics))
        mlflow.log_metric("total_drifted_features", total_drifted_features)
        mlflow.log_metric("drift_percentage", total_drifted_features / len(drift_metrics) * 100)
        for metric in drift_metrics:
            feature_key = metric['feature'].replace('/', '_per_').replace(' ', '_')
            mlflow.log_metric(f"{feature_key}_ks_statistic", metric["ks_statistic"])
            mlflow.log_metric(f"{feature_key}_p_value", metric["p_value"])
            mlflow.log_metric(f"{feature_key}_drift_detected", int(metric["drift_detected"]))
            

        logger.info("Drift report logged to MLflow")


if __name__ == "__main__":
    logger.info("Starting drift monitoring...")
    X_ref, y_ref, feature_names = load_reference_data()
    # Here you would load new data (X_new) that you want to compare against the reference data

    # For demonstration, we will just use the reference data as new data, which should show no drift
    X_new = np.load(BASE_DIR / "data" / "processed" / "X_test.npy")
    
    drift_metrics = compute_drift_metrics(X_ref, X_new, feature_names)
    log_drift_report(drift_metrics)
    logger.info("Drift monitoring completed")