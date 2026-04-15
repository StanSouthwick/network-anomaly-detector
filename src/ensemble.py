import joblib
import numpy as np
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
X_TEST_PATH = DATA_DIR / "X_test_scaled_production.npy"
Y_TEST_PATH = DATA_DIR / "y_test_production.npy"
IFOREST_MODEL_PATH = MODELS_DIR / "iforest_production_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_production_model.pkl"



def load_data(X_test_path=X_TEST_PATH, y_test_path=Y_TEST_PATH):
    if not all(path.exists() for path in [X_test_path, y_test_path]):
        raise FileNotFoundError("One or more test data files are missing. Please run the preprocessing step to generate the required files.")
        
    X_test = np.load(X_test_path)
    logger.info(f"Loaded X_test from {X_test_path} with shape {X_test.shape}")

    y_test = np.load(y_test_path)
    logger.info(f"Loaded y_test from {y_test_path} with shape {y_test.shape}")
        
    return X_test, y_test


class EnsembleModel:
    def __init__(self, iforest_model_path=IFOREST_MODEL_PATH, xgb_model_path=XGB_MODEL_PATH):
        self.iforest_model = joblib.load(iforest_model_path)
        self.xgb_model = joblib.load(xgb_model_path)
    

    def predict(self, X):
        y_pred_iforest = self.iforest_model.predict(X)
        y_pred_iforest = np.where(y_pred_iforest == -1, 1, 0)
        anomaly_flagged = y_pred_iforest == 1
        final_predictions = np.zeros(len(X), dtype=int)
        final_predictions[anomaly_flagged] = self.xgb_model.predict(X[anomaly_flagged])
        return final_predictions
    
    def evaluate(self, y_true, y_pred):
        conf_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        conf_report_string = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        logger.info("Classification Report:\n" + conf_report_string)
        logger.info("Confusion Matrix:\n" + str(conf_matrix))
        logger.info(f"F1 Score: {f1}")

        mlflow.set_experiment("network-anomaly-detector")
        with mlflow.start_run(run_name="EnsembleModel_Evaluation"):
            mlflow.log_metric("f1_score", f1)
            mlflow.log_dict(conf_report_dict, "classification_report.json")
            mlflow.log_dict({"confusion_matrix": conf_matrix.tolist()}, "confusion_matrix.json")
        return conf_report_dict, conf_matrix
    

if __name__ == "__main__":
    if not all(path.exists() for path in [X_TEST_PATH, Y_TEST_PATH]):
        raise FileNotFoundError("One or more test data files are missing. Please run the preprocessing step to generate the required files.")

    X_test, y_test = load_data()

    ensemble_model_production  = EnsembleModel()
    logger.info("Ensemble model loaded successfully.")

    y_pred = ensemble_model_production .predict(X_test)
    logger.info(f"Predictions made with shape {y_pred.shape}")

    ensemble_model_production .evaluate(y_test, y_pred) 
    logger.info("Ensemble model evaluation completed.")

   