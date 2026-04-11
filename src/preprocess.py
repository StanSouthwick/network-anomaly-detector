import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np

# log1p transformation removed — primary model (XGBoost) is tree-based and 
# unaffected by rescaling features that preserve their order. StandardScaler applied downstream.

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
DUPLICATE_COLUMN = "Fwd Header Length.1"  # Column from preprocessing notebook that is duplicated in the dataset
THRESHOLD_FOR_CORRELATION = 0.95
WEB_ATTACK_LABEL_FIXES = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection"
}
LABEL_COLUMN = "Label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(data_dir):
    # Load the dataset
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist.")
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    logger.info(f"Loading data from {data_dir}...")
    csv_files = list(data_dir.glob("*.csv")) # List all CSV files in the raw data directory
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}.")
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")
        
    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    # Concatenate all DataFrames into one
    network_data = pd.concat(dataframes, ignore_index=True)
    logger.info(f"All CSVs loaded and concatenated. Total records: {len(network_data)}")
    return network_data


def preprocess_data(network_data):
    logger.info("Starting data preprocessing...")
    network_data.columns = network_data.columns.str.strip()  # Remove leading and trailing whitespace
    logger.info("Column names stripped of whitespace.")
    return network_data

def drop_duplicate_rows(network_data):
    initial_count = len(network_data)
    logger.info(f"Initial record count before dropping duplicates: {initial_count}")
    network_data = network_data.drop_duplicates()
    final_count = len(network_data)
    logger.info(f"Dropped {initial_count - final_count} duplicate rows. Remaining records: {final_count}")
    return network_data

def drop_duplicate_columns(network_data, column_name):
    if column_name in network_data.columns:
        network_data = network_data.drop(columns=[column_name])
        logger.info(f"Dropped duplicate column: {column_name}")
    else:
        logger.warning(f"Column '{column_name}' not found in the dataset. No columns dropped.")
    return network_data

def replace_inf_with_nan(network_data):
    # Replace infinite values with NaN
    network_data =  network_data.replace([float('inf'), float('-inf')], pd.NA)
    logger.info("Replaced infinite values with NaN.")
    return network_data

def drop_missing_values(network_data):
    initial_count = len(network_data)
    network_data = network_data.dropna()
    final_count = len(network_data)
    logger.info(f"Dropped {initial_count - final_count} records with missing values. Remaining records: {final_count}")
    return network_data

def drop_zero_variance(network_data):
    # Drop columns with zero variance
    no_variance = network_data.var(numeric_only=True) == 0
    zero_variance_columns = no_variance[no_variance].index.tolist()
    network_data = network_data.drop(columns=zero_variance_columns)
    logger.info(f"Dropped {len(zero_variance_columns)} zero-variance columns: {zero_variance_columns}")
    return network_data


def drop_highly_correlated(network_data, threshold):
    correlations = network_data.select_dtypes(include='number').corr() # Calculate the correlation matrix for numeric columns
    # Identify pairs of columns with high correlation
    high_corr = (correlations.abs() > threshold) & (correlations.abs() < 1.0)
    pairs = [(col, row) for col in correlations.columns 
         for row in correlations.index 
         if high_corr.loc[row, col] and row < col]

    columns_to_drop = set()
    # Print the pairs of highly correlated columns
    for col1, col2 in pairs:
        if col2 not in columns_to_drop:
            columns_to_drop.add(col2)
    logger.info(f"\nColumns to drop due to high correlation: {columns_to_drop}")
    network_data = network_data.drop(columns=columns_to_drop)
    return network_data

def replace_encoding_issues(network_data, label_column, replacements):
    for column, replacement in replacements.items():
        network_data[label_column] = network_data[label_column].replace(column, replacement)
    logger.info(f"Replaced encoding issues in column '{label_column}' with specified replacements.")
    return network_data

def fit_encoder(network_data, label_column):
    label_encoder_production = LabelEncoder()
    network_data[label_column] = label_encoder_production.fit_transform(network_data[label_column])
    logger.info("Fitted label encoder to categorical column.")

    if not MODELS_DIR.exists(): # Ensure the models directory exists before saving the label encoder
        MODELS_DIR.mkdir(parents=True)
        logger.info(f"Created models directory at {MODELS_DIR}")

    joblib.dump(label_encoder_production, MODELS_DIR / "label_encoder_production.joblib")
    logger.info(f"Saved label encoder to {MODELS_DIR / 'label_encoder_production.joblib'}")
    
    
    return network_data

def train_test_split_data(network_data,  label_column, test_size, random_state):
    X = network_data.drop(columns=[label_column])
    y = network_data[label_column]
    X_train_production, X_test_production, y_train_production, y_test_production = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info(f"Split data into training and testing sets with test size {test_size} and random state {random_state}.")
    np.save(PROCESSED_DATA_DIR / "X_train_production.npy", X_train_production.to_numpy())
    np.save(PROCESSED_DATA_DIR / "X_test_production.npy", X_test_production.to_numpy())
    np.save(PROCESSED_DATA_DIR / "y_train_production.npy", y_train_production.to_numpy())
    np.save(PROCESSED_DATA_DIR / "y_test_production.npy", y_test_production.to_numpy())
    logger.info(f"Saved training and testing data to {PROCESSED_DATA_DIR} as NumPy arrays.")
    return X_train_production, X_test_production, y_train_production, y_test_production

def scale_features(X_train, X_test):
    network_data_scaler_production = Pipeline([
        ('scaler', StandardScaler())
    ])
    logger.info("Initialized feature scaler pipeline.")
    X_train_scaled_production = network_data_scaler_production.fit_transform(X_train)
    X_test_scaled_production = network_data_scaler_production.transform(X_test)
    logger.info("Scaled training and testing features using the scaler pipeline.")

    if not MODELS_DIR.exists(): # Ensure the models directory exists before saving the scaler pipeline
        MODELS_DIR.mkdir(parents=True)
        logger.info(f"Created models directory at {MODELS_DIR}")
    
    if not PROCESSED_DATA_DIR.exists(): # Ensure the processed data directory exists before saving scaled features
        PROCESSED_DATA_DIR.mkdir(parents=True)
        logger.info(f"Created processed data directory at {PROCESSED_DATA_DIR}")

    joblib.dump(network_data_scaler_production, MODELS_DIR / "network_data_scaler_production.joblib")
    logger.info(f"Saved feature scaler pipeline to {MODELS_DIR / 'network_data_scaler_production.joblib'}")
    np.save(PROCESSED_DATA_DIR / "X_train_scaled_production.npy", X_train_scaled_production)
    logger.info(f"Saved scaled training features to {PROCESSED_DATA_DIR / 'X_train_scaled_production.npy'}")
    np.save(PROCESSED_DATA_DIR / "X_test_scaled_production.npy", X_test_scaled_production)
    logger.info(f"Saved scaled testing features to {PROCESSED_DATA_DIR / 'X_test_scaled_production.npy'}")
    return X_train_scaled_production, X_test_scaled_production



if __name__ == "__main__":    
    network_data = load_data(RAW_DATA_DIR)
    network_data = preprocess_data(network_data)
    network_data = drop_duplicate_rows(network_data)
    network_data = drop_duplicate_columns(network_data, DUPLICATE_COLUMN)
    network_data = replace_inf_with_nan(network_data)
    network_data = drop_missing_values(network_data)
    network_data = drop_zero_variance(network_data)
    network_data = drop_highly_correlated(network_data, THRESHOLD_FOR_CORRELATION)
    network_data = replace_encoding_issues(network_data, LABEL_COLUMN, WEB_ATTACK_LABEL_FIXES)
    network_data = fit_encoder(network_data, LABEL_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split_data(network_data, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    logger.info("Data preprocessing completed successfully.")

    
