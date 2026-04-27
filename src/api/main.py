from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
from src.api.predict import load_models_and_artifacts, predict_classify, predict_anomaly, predict_ensemble, prepare_input_data
from src.api.schemas import PredictionResponse, FlowRecord, BatchRequest, BatchResponse

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ...

app = FastAPI(title="Network Anomaly Detector API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# Define API endpoints for anomaly detection, classification, ensemble prediction, batch prediction, and health check
@app.post("/predict/anomaly", response_model=PredictionResponse)
async def predict_anomaly_endpoint(request: FlowRecord):
    input_array = prepare_input_data(request, app.state.feature_names)
    input_array_scaled = app.state.scaler.transform(input_array)
    return predict_anomaly(input_array_scaled, app.state.iforest_model)

@app.post("/predict/classify", response_model=PredictionResponse)
async def classify(request: FlowRecord):
    input_array = prepare_input_data(request, app.state.feature_names)
    input_array_scaled = app.state.scaler.transform(input_array)
    return predict_classify(input_array_scaled, app.state.xgb_model, app.state.label_encoder)

@app.post("/predict/ensemble", response_model=PredictionResponse)
async def ensemble_predict(request: FlowRecord):
    input_array = prepare_input_data(request, app.state.feature_names)
    input_array_scaled = app.state.scaler.transform(input_array)
    return predict_ensemble(input_array_scaled, app.state.iforest_model, app.state.xgb_model, app.state.label_encoder)

@app.post("/predict/batch", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    responses = []
    for record in request.records:
        input_array = prepare_input_data(record, app.state.feature_names)
        input_array_scaled = app.state.scaler.transform(input_array)
        response = predict_ensemble(input_array_scaled, app.state.iforest_model, app.state.xgb_model, app.state.label_encoder)
        responses.append(response)
    return BatchResponse(predictions=responses)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": all([
            hasattr(app.state, "xgb_model"),
            hasattr(app.state, "iforest_model"),
            hasattr(app.state, "scaler"),
            hasattr(app.state, "label_encoder")
        ])
    }
@app.get("/")
async def root():
    return RedirectResponse(url="/health")
