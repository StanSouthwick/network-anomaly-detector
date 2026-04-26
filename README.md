# Network Anomaly Detector

![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-F1%3D0.9036-ff6600)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ed?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-ff9900?logo=amazonaws&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088ff?logo=githubactions&logoColor=white)

Production-grade ML pipeline for network intrusion detection, trained on the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset (2.83M labelled network flows). Demonstrates the full ML engineering lifecycle — from exploratory analysis through to a containerised inference API deployed on AWS, with automated CI/CD and runtime drift monitoring.

Built as a portfolio project targeting ML engineering roles in the defence and security sector.

---

## Architecture

```
  CICIDS2017 CSVs (8 files, ~2.83M rows)
          │
          ▼
  ┌───────────────┐
  │ preprocess.py │  Drop zero-variance, duplicate & high-correlation features
  │               │  48 features retained from 79 raw
  └───────┬───────┘
          │ artifacts/ (feature_names.json, scaler.pkl)
          ▼
  ┌───────────────────────────────────┐
  │  train_xgb.py                     │  XGBoost · sample_weight balancing
  │  train_anomaly.py                 │  Isolation Forest · BENIGN-only training
  └───────────────┬───────────────────┘
                  │ models/ (xgb.pkl, iso_forest.pkl)
                  ▼
  ┌───────────────────────────────────┐
  │  evaluate_xgb.py                  │  Classification report · confusion matrix
  │  evaluate_anomaly.py              │  Anomaly recall · contamination analysis
  │  ensemble.py                      │  IF → XGBoost pipeline · MLflow logging
  └───────────────┬───────────────────┘
                  │
          ┌───────┴────────┐
          │   S3 Bucket    │  Model artifacts stored; pulled at API startup
          └───────┬────────┘
                  │ boto3
                  ▼
  ┌───────────────────────────────────┐
  │  FastAPI Inference API            │
  │  POST /predict/classify           │
  │  POST /predict/anomaly            │
  │  POST /predict/ensemble           │
  │  POST /predict/batch              │
  └───────────────┬───────────────────┘
                  │
  ┌───────────────┴───────────────────┐
  │  Docker Container                 │
  └───────────────┬───────────────────┘
                  │
  ┌───────────────┴───────────────────┐
  │  GitHub Actions CI/CD             │
  │  push to main → build → ECR push  │
  │  → ECS Fargate redeploy           │
  └───────────────┬───────────────────┘
                  │
  ┌───────────────┴───────────────────┐
  │  Drift Monitoring                 │
  │  KS test per feature · MLflow     │
  │  src/monitor/drift.py             │
  └───────────────────────────────────┘
```

---

## Model Performance

| Model | Approach | Macro F1 |
|---|---|---|
| Random Forest v1 | SMOTE oversampling | 0.8785 |
| Random Forest v2 | `class_weight='balanced'` | 0.8696 |
| Keras DNN | Categorical cross-entropy | 0.3371 |
| **XGBoost** | **Sample weights (production)** | **0.9036** |
| Isolation Forest | Anomaly detection (BENIGN-only training) | — |
| IF → XGBoost Ensemble | Novelty gate + classifier | 0.87* |

*Ensemble macro F1 is lower than standalone XGBoost by design — Isolation Forest gates novel/unseen traffic that XGBoost was not trained on. The ensemble's value is breadth of detection, not benchmark maximisation.

---

## Key Design Decisions

**XGBoost with sample weights over SMOTE.** SMOTE introduces synthetic samples that can misrepresent minority-class decision boundaries. Sample weights achieve the same rebalancing effect during training without modifying the data distribution — preferable for a production setting where inference operates on raw flows.

**No log1p transformation.** Tree-based models are invariant to monotonic feature transformations. Log1p was evaluated and dropped deliberately; the notebook records this decision with supporting evidence.

**Isolation Forest as a supporting actor.** IF is trained exclusively on BENIGN traffic and used to flag flows that deviate from the normal operational envelope — not to maximise labelled-benchmark F1. Hyperparameter tuning was deliberately skipped: the model's role is novelty detection, and over-fitting its contamination parameter to CICIDS2017 labels would undermine that purpose.

**Ensemble logic: IF gates into XGBoost.** Anomaly-flagged records are routed to XGBoost for classification; non-flagged records bypass IF entirely. This preserves XGBoost's high precision on known attack types while extending coverage to out-of-distribution traffic.

**Preprocessing determinism.** Correlation-based feature dropping uses `np.triu(k=1)` to ensure a stable, reproducible column set regardless of execution order or string comparison behaviour.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML | XGBoost, scikit-learn (Isolation Forest, preprocessing) |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic v2 |
| Containerisation | Docker |
| Registry | AWS ECR |
| Compute | AWS ECS Fargate |
| Artifact storage | AWS S3 (boto3) |
| CI/CD | GitHub Actions |
| Drift monitoring | KS test · MLflow · `src/monitor/drift.py` |

---

## Project Structure

```
network-anomaly-detector/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory analysis, class distribution
│   ├── 02_preprocessing.ipynb        # Feature engineering decisions
│   ├── 03_model_training.ipynb       # Model comparison (RF, XGBoost, DNN)
│   └── 04_anomaly_detection.ipynb    # Isolation Forest, ensemble construction
├── src/
│   ├── preprocess.py                 # Feature selection, scaling, artifact export
│   ├── train_xgb.py                  # XGBoost training
│   ├── train_anomaly.py              # Isolation Forest training
│   ├── evaluate_xgb.py               # Classification evaluation
│   ├── evaluate_anomaly.py           # Anomaly evaluation
│   ├── ensemble.py                   # IF → XGBoost pipeline
│   ├── api/
│   │   ├── main.py                   # FastAPI app, lifespan, S3 artifact loading
│   │   ├── predict.py                # Endpoint logic
│   │   └── schemas.py                # Pydantic request/response models
│   └── monitor/
│       └── drift.py                  # KS-test drift detection
├── artifacts/                        # feature_names.json, scaler.pkl (git-ignored)
├── models/                           # .pkl model files (git-ignored, stored in S3)
├── dashboard.html                    # Portfolio demo (DEMO/LIVE mode)
├── Dockerfile
├── .github/workflows/deploy.yml      # CI/CD pipeline
└── requirements.txt
```

---

## Setup

### Local (scripts)

```bash
git clone https://github.com/StanSouthwick/network-anomaly-detector.git
cd network-anomaly-detector

conda create -n anomaly-detector python=3.12
conda activate anomaly-detector
pip install -r requirements.txt

# Download CICIDS2017 CSVs to data/raw/, then:
python src/preprocess.py
python src/train_xgb.py
python src/train_anomaly.py
python src/evaluate_xgb.py
python src/ensemble.py
```

### API (Docker)

```bash
docker build -t anomaly-detector .
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e S3_BUCKET=your-bucket-name \
  anomaly-detector
```

API available at `http://localhost:8000`. Interactive docs at `/docs`.

---

## API Endpoints

All endpoints accept a single network flow as a JSON object with 48 float fields matching the CICIDS2017 feature set (see `artifacts/feature_names.json`).

| Endpoint | Method | Description |
|---|---|---|
| `/predict/classify` | POST | XGBoost label + confidence |
| `/predict/anomaly` | POST | Isolation Forest anomaly flag + score |
| `/predict/ensemble` | POST | IF-gated XGBoost prediction |
| `/predict/batch` | POST | Batch inference (array of flows) |

Example request:

```bash
curl -X POST http://localhost:8000/predict/classify \
  -H "Content-Type: application/json" \
  -d @sample_flow.json
```

Example response:

```json
{
  "label": "DoS Hulk",
  "confidence": 0.994,
  "model": "xgboost"
}
```

---

## Drift Monitoring

`src/monitor/drift.py` computes per-feature Kolmogorov–Smirnov statistics between the CICIDS2017 training distribution and incoming inference traffic. Results are logged to MLflow. Thresholds: KS > 0.5 triggers a high-drift alert; 0.2–0.5 is moderate.

The portfolio dashboard (`dashboard.html`) visualises current KS statistics per feature with colour-coded severity.

---

## Data

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity. 2.83M labelled network flows across 15 traffic classes including DoS, DDoS, PortScan, Brute Force, Web Attacks, and Infiltration. Raw CSVs are not included in this repository.
