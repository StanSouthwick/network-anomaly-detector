import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("/Users/stans/eportfolio/eportfolio_Stanley-Southwick/network-anomaly-detector/data/raw")
DUPLICATE_COLUMN = "Fwd Header Length.1"  # Column from preprocessing notebook that is duplicated in the dataset

def load_data(data_dir):
    # Load the dataset
    
    csv_files = data_dir.glob("*.csv") # List all CSV files in the raw data directory
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name  # Add a column to identify the source file
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


if __name__ == "__main__":    
    network_data = load_data(RAW_DATA_DIR)
    network_data = preprocess_data(network_data)
    network_data = drop_duplicate_rows(network_data)
    network_data = drop_duplicate_columns(network_data, DUPLICATE_COLUMN)
    network_data = replace_inf_with_nan(network_data)
    network_data = drop_missing_values(network_data)
    
