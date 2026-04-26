import boto3
import joblib
import logging
import json
import numpy as np
from src.api.schemas import PredictionResponse

logger = logging.getLogger(__name__)
s3 = boto3.client('s3')

# Constants for S3 bucket and artifact keys
BUCKET = 'network-anomaly-detector-artifacts'
FEATURE_NAMES = 'artifacts/feature_names.json'
IFOREST_MODEL = 'models/iforest_production_model.pkl'
LABEL_ENCODER = 'models/label_encoder_production.joblib'
SCALER = 'models/network_data_scaler_production.joblib'
XGB_MODEL = 'models/xgb_production_model.pkl'
LOCAL_PATH = '/tmp/'

# Utility functions for loading models/artifacts and making predictions
def load_artifact_from_s3(bucket, key, local_path, model_name):
    local_file_path = local_path + key.split('/')[-1]
    try:
        s3.download_file(bucket, key, local_file_path)
        logger.info(f"Successfully downloaded {key} from S3 bucket {bucket}")
        if local_file_path.endswith('.json'):
            with open(local_file_path) as f:
                artifact = json.load(f)
        else:
            artifact = joblib.load(local_file_path)
        logger.info(f"Successfully loaded {model_name} from {local_file_path}")
        return artifact
        
    except Exception as e:
        logger.error(f"Error downloading {key} from S3 bucket {bucket}: {e}")
        raise e

# Functions for preparing input data and making predictions
def load_models_and_artifacts():
    logger.info("Loading models and artifacts from S3...")
    feature_names = load_artifact_from_s3(BUCKET, FEATURE_NAMES, LOCAL_PATH, "Feature Names")
    iforest_model = load_artifact_from_s3(BUCKET, IFOREST_MODEL, LOCAL_PATH, "Isolation Forest Model")
    label_encoder = load_artifact_from_s3(BUCKET, LABEL_ENCODER, LOCAL_PATH, "Label Encoder")
    scaler = load_artifact_from_s3(BUCKET, SCALER, LOCAL_PATH, "Scaler")
    xgb_model = load_artifact_from_s3(BUCKET, XGB_MODEL, LOCAL_PATH, "XGBoost Model")
    logger.info("All models and artifacts loaded successfully")
    
    return feature_names, iforest_model, label_encoder, scaler, xgb_model

# Function to prepare input data for prediction, ensuring correct feature order and handling aliases
def prepare_input_data(record, feature_names):
    input_data = []
    record_by_alias = record.model_dump(by_alias=True)
    record_by_field = record.model_dump()

    for feature in feature_names:
        if feature in record_by_alias:
            input_data.append(record_by_alias[feature])
        elif feature.replace(' ', '_') in record_by_field:
            input_data.append(record_by_field[feature.replace(' ', '_')])
        else:
            raise ValueError(f"Missing feature: {feature}")

    logger.info(f"Prepared input data for prediction successfully")
    return np.array(input_data).reshape(1, -1)

# Prediction functions for classification, anomaly detection, and ensemble prediction
def predict_classify(input_array, xgb_model, label_encoder):
    try:
        prediction = xgb_model.predict(input_array)[0]
        confidence = float(np.max(xgb_model.predict_proba(input_array)))
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence}")
        return PredictionResponse(prediction=prediction_label, confidence=confidence, anomaly_flagged=False)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e

# Anomaly detection function using Isolation Forest, returning a PredictionResponse with anomaly flag
def predict_anomaly(input_array, iforest_model):
    try:
        anomaly_score = iforest_model.decision_function(input_array)[0]
        is_anomaly = anomaly_score < 0
        logger.info(f"Anomaly Score: {anomaly_score}, Anomaly Flagged: {is_anomaly}")
        return PredictionResponse(prediction="UNKNOWN", confidence=0.0, anomaly_flagged=is_anomaly)
    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}")
        raise e

# Ensemble prediction function that combines results from both anomaly detection and classification, 
# returning a PredictionResponse with the final prediction, confidence, and anomaly flag
def predict_ensemble(input_array_scaled, iforest_model, xgb_model, label_encoder):
    anomaly_response = predict_anomaly(input_array_scaled, iforest_model)
    is_anomaly = anomaly_response.anomaly_flagged
    classify_response = predict_classify(input_array_scaled, xgb_model, label_encoder)
    return PredictionResponse(prediction=classify_response.prediction, confidence=classify_response.confidence, anomaly_flagged=is_anomaly)