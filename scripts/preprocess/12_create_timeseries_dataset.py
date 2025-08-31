
# -*- coding: utf-8 -*-
"""
Prepares and transforms data into time-series format for deep learning models.

This script takes the split datasets (train, validation, test) and the raw
event data (labs, medications) to create fixed-length sequences suitable for
models like Temporal Fusion Transformer (TFT) or GRU-D.

The output is three Parquet files containing the time-series data, which include:
- Static features (e.g., demographics)
- Time-varying known inputs (e.g., medication schedules)
- Time-varying observed inputs (e.g., lab results)
- The true outcome label
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"
RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_PATH = PROCESSED_DATA_PATH

# Input files
SPLIT_FILES = {
    "train": PROCESSED_DATA_PATH / "train_dataset.json",
    "validation": PROCESSED_DATA_PATH / "validation_dataset.json",
    "test": PROCESSED_DATA_PATH / "test_dataset.json",
}
OUTCOMES_FILE = PROCESSED_DATA_PATH / "outcomes.json"
RAW_FILES = {
    "lab": RAW_DATA_PATH / "LIS 去除身份证.xlsx",
    "inpatient": RAW_DATA_PATH / "HIS住院.xlsx",
}

# Parameters
SEQUENCE_LENGTH = 72  # 72 hours

def sanitize_column_name(name):
    """Sanitizes column names to be valid for PyTorch Forecasting."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

def create_timeseries_dataset():
    """
    Main function to orchestrate the time-series data creation.
    """
    print("Starting time-series dataset creation...")

    # --- 1. Load All Necessary Data ---
    print("Loading datasets, outcomes, and raw event data...")
    try:
        with open(OUTCOMES_FILE, 'r', encoding='utf-8') as f:
            outcomes = json.load(f)

        with open(PROCESSED_DATA_PATH / "analysis_ready_data.json", 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        full_df = pd.DataFrame(full_data)
        full_df['inpatient_id'] = full_df['inpatient_id'].astype(str)


        split_ids = {}
        for key, path in SPLIT_FILES.items():
            with open(path, 'r') as f:
                split_ids[key] = json.load(f)

        split_dfs = {
            key: full_df[full_df['inpatient_id'].isin(ids)]
            for key, ids in split_ids.items()
        }

        raw_dfs = {key: pd.read_excel(path) for key, path in RAW_FILES.items()}
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # --- 2. Pre-process Raw Data ---
    # Standardize IDs and parse dates for raw event data
    lab_df = raw_dfs['lab']
    lab_df.rename(columns={'INPATIENT_ID': 'inpatient_id'}, inplace=True)
    lab_df['inpatient_id'] = lab_df['inpatient_id'].astype(str)
    lab_df['timestamp'] = pd.to_datetime(lab_df['INSPECTION_DATE'], errors='coerce')
    lab_df['value'] = pd.to_numeric(lab_df['QUANTITATIVE_RESULT'], errors='coerce')
    lab_df.dropna(subset=['timestamp', 'value', 'CHINESE_NAME'], inplace=True)
    
    # Create a mapping for unique sanitized lab test names
    lab_test_names = lab_df['CHINESE_NAME'].unique()
    lab_col_map = {name: f'lab_{sanitize_column_name(name)}_{i}' for i, name in enumerate(lab_test_names)}
    lab_df['sanitized_lab_name'] = lab_df['CHINESE_NAME'].map(lab_col_map)
    
    lab_df_pivot = lab_df.pivot_table(
        index=['inpatient_id', 'timestamp'],
        columns='sanitized_lab_name',
        values='value'
    ).reset_index()
    lab_df_pivot.columns = [sanitize_column_name(col) for col in lab_df_pivot.columns]


    inpatient_df = raw_dfs['inpatient']
    inpatient_df.rename(columns={'住院号码': 'inpatient_id'}, inplace=True)
    inpatient_df['inpatient_id'] = inpatient_df['inpatient_id'].astype(str)
    inpatient_df['timestamp'] = pd.to_datetime(inpatient_df['医嘱开始时间'], errors='coerce')
    inpatient_df.dropna(subset=['timestamp', '药品医嘱'], inplace=True)
    
    # Create a mapping for unique sanitized column names
    top_meds = inpatient_df['药品医嘱'].value_counts().nlargest(30).index
    med_col_map = {med: f'med_{sanitize_column_name(med)}_{i}' for i, med in enumerate(top_meds)}
    
    med_cols = []
    for original_name, new_name in med_col_map.items():
        inpatient_df[new_name] = (inpatient_df['药品医嘱'] == original_name).astype(int)
        med_cols.append(new_name)

    all_lab_cols = lab_df_pivot.columns.drop(['inpatient_id', 'timestamp']).tolist()
    all_med_cols = med_cols

    # --- 3. Process Each Split (Train, Val, Test) ---
    for split_name, df in split_dfs.items():
        print(f"\nProcessing '{split_name}' split...")
        all_patient_sequences = []

        # Add true outcome to the dataframe, filling missing outcomes with 0
        df['outcome'] = df['inpatient_id'].map(outcomes).fillna(0)
        df['t0_admission_time'] = pd.to_datetime(df['t0_admission_time'])

        for _, patient_row in df.iterrows():
            patient_id = patient_row['inpatient_id']
            t0 = patient_row['t0_admission_time']
            
            # Create the base sequence DataFrame for this patient
            time_index = pd.to_datetime([t0 + pd.Timedelta(hours=i) for i in range(SEQUENCE_LENGTH)])
            sequence_df = pd.DataFrame({'time_idx': range(SEQUENCE_LENGTH)}, index=time_index)
            sequence_df['inpatient_id'] = patient_id

            # --- Add Static Features ---
            sequence_df['age'] = patient_row['demographics'].get('age', 0)
            sequence_df['sex'] = patient_row['demographics'].get('sex', 'Unknown')

            # --- Add Time-Varying Observed Inputs (Labs) ---
            patient_labs = lab_df_pivot[lab_df_pivot['inpatient_id'] == patient_id].copy()
            if not patient_labs.empty:
                patient_labs['hour'] = (patient_labs['timestamp'] - t0).dt.total_seconds() // 3600
                patient_labs = patient_labs[(patient_labs['hour'] >= 0) & (patient_labs['hour'] < SEQUENCE_LENGTH)]
                lab_features = patient_labs.drop(columns=['inpatient_id', 'timestamp']).groupby('hour').mean()
                
                # Resample to hourly and merge
                base_lab_df = pd.DataFrame(index=range(SEQUENCE_LENGTH))
                merged_labs = base_lab_df.merge(lab_features, left_index=True, right_index=True, how='left')
                merged_labs = merged_labs.reindex(columns=all_lab_cols)
                merged_labs.ffill(inplace=True) # Forward-fill missing values
                merged_labs.fillna(0, inplace=True) # Fill remaining NaNs with 0
                
                for col in merged_labs.columns:
                    sequence_df[sanitize_column_name(col)] = merged_labs[col].values

            # --- Add Time-Varying Known Inputs (Meds) ---
            patient_meds = inpatient_df[inpatient_df['inpatient_id'] == patient_id][['timestamp'] + med_cols]
            if not patient_meds.empty:
                patient_meds['hour'] = (patient_meds['timestamp'] - t0).dt.total_seconds() // 3600
                patient_meds = patient_meds[(patient_meds['hour'] >= 0) & (patient_meds['hour'] < SEQUENCE_LENGTH)]
                med_features = patient_meds.drop(columns=['timestamp']).groupby('hour').max() # Get max if multiple orders in same hour
                
                base_med_df = pd.DataFrame(index=range(SEQUENCE_LENGTH))
                merged_meds = base_med_df.merge(med_features, left_index=True, right_index=True, how='left')
                merged_meds = merged_meds.reindex(columns=all_med_cols)
                merged_meds.fillna(0, inplace=True)

                for col in merged_meds.columns:
                    sequence_df[sanitize_column_name(col)] = merged_meds[col].values

            # --- Add Target ---
            sequence_df['outcome'] = patient_row['outcome']
            
            all_patient_sequences.append(sequence_df)

        if not all_patient_sequences:
            print(f"  - No sequences created for '{split_name}' split. Skipping file save.")
            continue
            
        # Concatenate all patient sequences into one big DataFrame
        final_df = pd.concat(all_patient_sequences, ignore_index=True)
        
        # Final fillna to catch any remaining missing values
        final_df.fillna(0, inplace=True)
        
        # Clean up column names for Parquet
        final_df.columns = [sanitize_column_name(col) for col in final_df.columns]
        
        # --- 4. Save the Final Time-Series DataFrame ---
        output_file = OUTPUT_PATH / f"{split_name}_timeseries.parquet"
        print(f"  - Saving {len(final_df)} records to {output_file}...")
        final_df.to_parquet(output_file, index=False)

    print("\nTime-series dataset creation complete.")


if __name__ == "__main__":
    create_timeseries_dataset()
