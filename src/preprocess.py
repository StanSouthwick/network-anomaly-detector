import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("/Users/stans/eportfolio/eportfolio_Stanley-Southwick/network-anomaly-detector/data/raw")

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


if __name__ == "__main__":    
    network_data = load_data(RAW_DATA_DIR)
    
