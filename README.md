# Network-anomaly-detctor

A production-grade machine learning system for detecting anomalous network traffic, built on the CICIDS2017 dataset, a modern, realistic benchmark containing ~2.8 million labelled network flows across 79 features spanning both benign and attack traffic.

This project covers the full ML engineering lifecycle: exploratory data analysis, preprocessing and feature selection, model training and evaluation, production-ready inference API, containerisation, and data drift monitoring. The goal is to  demonstrate the practices that separate research notebooks from production systems such as structured code, proper error handling, meaningful version control, and reproducible pipelines.

A hybrid approach is used, combining traditional ensemble methods (XGBoost, Random Forest, Isolation Forest) with a deep learning baseline (TensorFlow/Keras), enabling direct comparison across paradigms for this class of problem.

Tech stack: Python // scikit-learn // XGBoost // TensorFlow/Keras // FastAPI // Docker // MLflow

## Dataset
CIC-IDS-2017 - Machine Learning CSV : https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FCSVs 
The databse chosen 'CIDIDS2017' contains 2.8 million rows and is sperated across 8 files for different times of the working weeks representing 5 days of enwtork traffic with 14 different attack types and 79 feature.

## Progress

### Phase 1 — Exploratory Data Analysis ✅
Documented key dataset characteristics across all 8 CSVs:
- ~9% duplicate rows
- Duplicate column (`Fwd Header Length.1`)
- Infinite values in `Flow Bytes/s` and `Flow Packets/s` 
- 8 zero-variance features
- 33 highly correlated feature pairs (above 0.95 threshold)
- Character encoding errors in Web Attack labels
- Severe class imbalance (~80% BENIGN)

### Phase 2 — Preprocessing & Feature Engineering ✅
Applied all fixes identified in Phase 1. Key decisions:
- Infinite/null rows in flow rate columns were **dropped** rather than imputed — underlying records are likely corrupt (division by zero), and imputation would introduce synthetic values for fundamentally invalid data
- Zero-variance and highly correlated features removed programmatically (not hardcoded)
- Web Attack labels cleaned via `str.replace`
- Labels encoded with `LabelEncoder` (saved to `models/`)
- Stratified 80/20 train/test split to preserve class distribution
- Feature scaling via `StandardScaler` wrapped in sklearn `Pipeline` (saved to `models/`)
- Final dataset: 2,520,798 rows, 51 features
- SMOTE applied to `X_train` only to oversample rare attack classes — prioritises reducing false negatives (missed attacks) over the risk of synthetic data, which is the correct trade-off in a security context

### Phase 3 — Model Training & Evaluation 🔄


## Project structure

## Setup
