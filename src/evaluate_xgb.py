from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import numpy as np
import logging
from pathlib import Path
import mlflow


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
mlflow.set_experiment("network-anomaly-detector")
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
XGB_MODEL_DIR = BASE_DIR / "models" / "xgb_production_model.pkl"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_scaled_production.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test_production.npy"

def load_data():
    if not all(path.exists() for path in [X_TEST_PATH, Y_TEST_PATH]):
        raise FileNotFoundError("One or more processed data files are missing. Please run the preprocessing step to generate the required files.")
    # Load the preprocessed testing data
    X_test = np.load(X_TEST_PATH)
    logger.info(f"Loaded X_test from {X_TEST_PATH} with shape {X_test.shape}")

    y_test = np.load(Y_TEST_PATH)
    logger.info(f"Loaded y_test from {Y_TEST_PATH} with shape {y_test.shape}")

    
    return X_test, y_test

def evaluate_model(xgb_production_model, X_test, y_test):
    encoder = joblib.load(BASE_DIR / "models" / "label_encoder_production.joblib")
    y_pred = xgb_production_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=encoder.classes_)
    classification_report_str = classification_report(y_test, y_pred, target_names=encoder.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # MLFlow tracking
    

    with mlflow.start_run(run_name="XGBoost_production",):
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", classification_report_dict['macro avg']['precision'])
        mlflow.log_metric("recall", classification_report_dict['macro avg']['recall'])
        mlflow.log_dict(classification_report_dict, "classification_report.json")
        mlflow.log_dict({"confusion_matrix": conf_matrix.tolist()}, "confusion_matrix.json")
        

    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("Classification Report:\n" + classification_report_str)
    logger.info("Confusion Matrix:\n" + str(conf_matrix))

    return f1, classification_report_str, classification_report_dict, conf_matrix



if __name__ == "__main__":
    if not XGB_MODEL_DIR.exists():
        raise FileNotFoundError(f"XGBoost model file not found at {XGB_MODEL_DIR}. Please run the training step to generate the model.")
    
    xgb_model = joblib.load(XGB_MODEL_DIR)
    logger.info(f"Loaded XGBoost model from {XGB_MODEL_DIR}")

    X_test, y_test = load_data()
    evaluate_model(xgb_model, X_test, y_test)
    logger.info("Evaluation completed successfully.")