
import pandas as pd
import os
import json
import glob
from sklearn.preprocessing import MinMaxScaler

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
UNIFIED_FILE = os.path.join(PROCESSED_DIR, 'unified_dataset.json')

# --- Data Loading ---
def load_processed_data(directory):
    """Loads all JSON files from the processed_data directory."""
    json_files = glob.glob(os.path.join(directory, '*.json'))
    data = {}
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            # Add a check for empty or invalid JSON
            try:
                content = json.load(f)
                data[file_name] = content
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_name}. Skipping.")
                continue
    return data

# --- Numerical Processing ---
def normalize_numerical_data(df):
    """Identifies numerical columns and applies Min-Max scaling."""
    scaler = MinMaxScaler()
    # Select only numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        # Important: handle potential missing values before scaling
        df[numeric_cols] = df[numeric_cols].fillna(0) 
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# --- Unification ---
def unify_data(all_data):
    """
    Unifies data from different sources into a structured format.
    """
    unified_records = []
    # Flatten all records and add a source file identifier
    for file_name, records in all_data.items():
        # Ensure records is a list of dicts
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    record['source_file'] = file_name
                    unified_records.append(record)

    # Convert to DataFrame for easier numerical processing
    df = pd.DataFrame(unified_records)
    
    # Normalize numerical data
    df = normalize_numerical_data(df)

    # --- PII Removal ---
    if 'patient_name' in df.columns:
        df.drop(columns=['patient_name'], inplace=True)
    
    return df.to_dict('records')


def main():
    """
    Main function to load, unify, clean, and save the dataset.
    """
    print("Loading processed data...")
    all_data = load_processed_data(PROCESSED_DIR)
    
    print("Unifying and cleaning data...")
    unified_data = unify_data(all_data)
    
    print(f"Saving unified dataset to {UNIFIED_FILE}...")
    with open(UNIFIED_FILE, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=4)
        
    print("Processing complete.")

if __name__ == "__main__":
    main()
