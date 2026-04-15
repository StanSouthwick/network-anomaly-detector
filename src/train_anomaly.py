import numpy as np
import joblib
import mlflow
import logging
from sklearn.ensemble import IsolationForest
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)    


BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_scaled_production.npy"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train_production.npy"

def load_data():
    if not all(path.exists() for path in [X_TRAIN_PATH, Y_TRAIN_PATH]):
        raise FileNotFoundError("One or more processed data files are missing. Please run the preprocessing step to generate the required files.")
    # Load the preprocessed training data
    X_train = np.load(X_TRAIN_PATH)
    logger.info(f"Loaded X_train from {X_TRAIN_PATH} with shape {X_train.shape}")
    
    y_train = np.load(Y_TRAIN_PATH)
    logger.info(f"Loaded y_train from {Y_TRAIN_PATH} with shape {y_train.shape}")

    benign_labels = y_train[y_train == 0]
    logger.info(f"Extracted benign labels with shape {benign_labels.shape}")
    benign_traffic = X_train[y_train == 0]
    logger.info(f"Extracted benign traffic with shape {benign_traffic.shape}")

    joblib.dump(benign_traffic, MODELS_DIR / "iforest_production_benign_traffic.npy")
    joblib.dump(benign_labels, MODELS_DIR / "iforest_production_benign_labels.npy")
    logger.info(f"Saved benign traffic and labels for Isolation Forest training.")
    
    return X_train, y_train, benign_traffic, benign_labels

def train_isolation_forest(y_train, benign_traffic):
    
    # Using the full labelled dataset before the train/test split
    attack_proportion = (y_train != 0).sum() / len(y_train)
    contamination_production = float(min(attack_proportion, 0.5))  # Set contamination to the attack proportion, capped at 0.5

    # Save the calculated contamination for later use
    joblib.dump(contamination_production, MODELS_DIR / "iforest_production_contamination.pkl")
    logger.info(f"Calculated contamination for Isolation Forest: {contamination_production:.4f}")

    # Train the Isolation Forest model with the calculated contamination
    iforest_production_model = IsolationForest(n_estimators=100, contamination=contamination_production, random_state=42)
    iforest_production_model.fit(benign_traffic)

 # MLFlow tracking
    mlflow.set_experiment("network-anomaly-detector")
    with mlflow.start_run(run_name="IsolationForest_production",):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("contamination", contamination_production)
        mlflow.sklearn.log_model(iforest_production_model, name="model")

    logger.info("Trained Isolation Forest model on benign traffic.")
    joblib.dump(iforest_production_model, MODELS_DIR / "iforest_production_model.pkl")

if __name__ == "__main__":
    X_train, y_train, benign_traffic, benign_labels = load_data()
    train_isolation_forest(y_train, benign_traffic)
    logger.info("Isolation Forest training completed and model saved.")