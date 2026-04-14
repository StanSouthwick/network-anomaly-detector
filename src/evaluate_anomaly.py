import numpy as np
import joblib
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import mlflow
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_scaled_production.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test_production.npy"
IFOREST_MODEL_PATH = MODELS_DIR / "iforest_production_model.pkl"

def load_data():
    if not all(path.exists() for path in [X_TEST_PATH, Y_TEST_PATH]):
        raise FileNotFoundError("One or more processed data files are missing. Please run the preprocessing step to generate the required files.")
    # Load the preprocessed testing data
    X_test = np.load(X_TEST_PATH)
    logger.info(f"Loaded X_test from {X_TEST_PATH} with shape {X_test.shape}")

    y_test = np.load(Y_TEST_PATH)
    logger.info(f"Loaded y_test from {Y_TEST_PATH} with shape {y_test.shape}")


    
    return X_test, y_test


def evaluate_model(iforest_production_model, X_test, y_test):

    y_pred = iforest_production_model.predict(X_test)
    y_test_converted = np.where(y_test == 0, 1, -1)

    f1 = f1_score(y_test_converted, y_pred, average='macro')
    classification_report_dict = classification_report(y_test_converted, y_pred, output_dict=True)
    classification_report_str = classification_report(y_test_converted, y_pred)
    conf_matrix = confusion_matrix(y_test_converted, y_pred)

    # MLFlow tracking
    mlflow.set_experiment("network-anomaly-detector")

    with mlflow.start_run(run_name="IsolationForest_production",):
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", classification_report_dict['macro avg']['precision'])
        mlflow.log_metric("recall", classification_report_dict['macro avg']['recall'])

    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("Classification Report:\n" + classification_report_str)
    logger.info("Confusion Matrix:\n" + str(conf_matrix))

    return f1, classification_report_str, conf_matrix


if __name__ == "__main__":
    if not IFOREST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Isolation Forest model file not found at {IFOREST_MODEL_PATH}. Please run the training step to generate the model.")
    
    iforest_model = joblib.load(IFOREST_MODEL_PATH)
    logger.info(f"Loaded Isolation Forest model from {IFOREST_MODEL_PATH}")

    X_test, y_test = load_data()
    evaluate_model(iforest_model, X_test, y_test)
    logger.info("Evaluation completed successfully.")






