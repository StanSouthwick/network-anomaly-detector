import numpy as np
import joblib
import mlflow
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_scaled_production.npy"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train_production.npy"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_scaled_production.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test_production.npy"



def load_data():
    if not all(path.exists() for path in [X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH]):
        raise FileNotFoundError("One or more processed data files are missing. Please run the preprocessing step to generate the required files.")
    # Load the preprocessed training and testing data
    X_train = np.load(X_TRAIN_PATH)
    logger.info(f"Loaded X_train from {X_TRAIN_PATH} with shape {X_train.shape}")

    y_train = np.load(Y_TRAIN_PATH)
    logger.info(f"Loaded y_train from {Y_TRAIN_PATH} with shape {y_train.shape}")

    X_test = np.load(X_TEST_PATH)
    logger.info(f"Loaded X_test from {X_TEST_PATH} with shape {X_test.shape}")

    y_test = np.load(Y_TEST_PATH)
    logger.info(f"Loaded y_test from {Y_TEST_PATH} with shape {y_test.shape}")

    
    return X_train, y_train, X_test, y_test

def train_XGBoost_model(X_train, y_train):
    encoder = joblib.load(MODELS_DIR / "label_encoder_production.joblib")
    sample_weights = compute_sample_weight(class_weight='balanced', y = y_train)
    xgb_production_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', objective='multi:softmax', num_class=len(encoder.classes_))
    xgb_production_model.fit(X_train, y_train, sample_weight=sample_weights)

    # MLFlow tracking
    mlflow.set_experiment("network-anomaly-detector")
    with mlflow.start_run(run_name="XGBoost_production",):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.xgboost.log_model(xgb_production_model, name="model")

        logger.info("Trained XGBoost model with balanced sample weights.")
        joblib.dump(xgb_production_model, MODELS_DIR / "xgb_production_model.pkl")
        return xgb_production_model




if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    xgb_production_model = train_XGBoost_model(X_train, y_train)
    
    