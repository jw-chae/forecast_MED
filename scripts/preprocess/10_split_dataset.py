
# -*- coding: utf-8 -*-
"""
Splits the analysis-ready dataset into training, validation, and test sets
using stratified sampling to maintain outcome distribution.

This script ensures that data from the same patient (inpatient_id) does not
appear in more than one set, which is crucial for preventing data leakage.
It loads the real outcomes and performs a stratified split based on them.
"""
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"
INPUT_FILE = PROCESSED_DATA_PATH / "analysis_ready_data.json"
OUTCOMES_FILE = PROCESSED_DATA_PATH / "outcomes.json"
OUTPUT_TRAIN_FILE = PROCESSED_DATA_PATH / "train_dataset.json"
OUTPUT_VALIDATION_FILE = PROCESSED_DATA_PATH / "validation_dataset.json"
OUTPUT_TEST_FILE = PROCESSED_DATA_PATH / "test_dataset.json"

# Splitting ratios
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42  # for reproducibility

def split_dataset():
    """
    Loads the dataset, merges it with real outcomes, and performs a
    stratified split into training, validation, and test sets by patient ID.
    """
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}. Please run the preprocessing script first.")
        return

    if df.empty:
        print("Error: The dataset is empty. Cannot perform split.")
        return

    # --- Load and Merge Real Outcomes ---
    print(f"Loading real outcomes from {OUTCOMES_FILE}...")
    try:
        with open(OUTCOMES_FILE, 'r') as f:
            outcomes = json.load(f)
        outcomes_df = pd.DataFrame(list(outcomes.items()), columns=['inpatient_id', 'outcome'])
        outcomes_df['inpatient_id'] = outcomes_df['inpatient_id'].astype(str) # Ensure consistent type
    except FileNotFoundError:
        print(f"Error: Outcomes file not found at {OUTCOMES_FILE}. Please run the outcome generation script first.")
        return

    # Merge outcomes with the main dataframe
    df = pd.merge(df, outcomes_df, on='inpatient_id', how='left')
    df['outcome'].fillna(0, inplace=True) # Assume missing outcomes are negative
    df['outcome'] = df['outcome'].astype(int)

    # --- Stratified Split by Patient ID ---
    print("Performing stratified split by 'inpatient_id' to prevent data leakage...")
    
    # Create a dataframe with one row per patient and their outcome for stratification
    patient_outcomes = df.drop_duplicates(subset='inpatient_id').set_index('inpatient_id')['outcome']
    patient_ids = patient_outcomes.index
    labels = patient_outcomes.values

    # First split: separate test set
    train_val_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Second split: separate validation set from the remaining data
    train_val_labels = patient_outcomes[train_val_ids]
    relative_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=train_val_labels
    )

    # Filter the DataFrame to create the final sets
    train_df = df[df['inpatient_id'].isin(train_ids)]
    validation_df = df[df['inpatient_id'].isin(val_ids)]
    test_df = df[df['inpatient_id'].isin(test_ids)]

    print("\nDataset split complete:")
    print(f"  - Training set:   {len(train_df)} records ({len(train_ids)} patients)")
    print(f"  - Validation set: {len(validation_df)} records ({len(val_ids)} patients)")
    print(f"  - Test set:       {len(test_df)} records ({len(test_ids)} patients)")

    # --- Save the Datasets ---
    # We need to save only the patient IDs, not the full data, to match the original pipeline
    print("\nSaving split patient ID lists to JSON files...")
    with open(OUTPUT_TRAIN_FILE, 'w') as f:
        json.dump(train_ids.tolist(), f)
    with open(OUTPUT_VALIDATION_FILE, 'w') as f:
        json.dump(val_ids.tolist(), f)
    with open(OUTPUT_TEST_FILE, 'w') as f:
        json.dump(test_ids.tolist(), f)

    print(f"  - Saved training set patient IDs to {OUTPUT_TRAIN_FILE}")
    print(f"  - Saved validation set patient IDs to {OUTPUT_VALIDATION_FILE}")
    print(f"  - Saved test set patient IDs to {OUTPUT_TEST_FILE}")
    print("  - Done.")

if __name__ == "__main__":
    split_dataset()
