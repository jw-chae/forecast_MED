

import pandas as pd
from pathlib import Path
import json

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"
TRAIN_FILE = PROCESSED_DATA_PATH / "train_dataset.json"
VALIDATION_FILE = PROCESSED_DATA_PATH / "validation_dataset.json"
TEST_FILE = PROCESSED_DATA_PATH / "test_dataset.json"
OUTCOMES_FILE = PROCESSED_DATA_PATH / "outcomes.json"

def inspect_split_distribution():
    """
    Loads the split datasets and inspects the outcome distribution.
    """
    print("--- Inspecting Data Split Distribution ---")

    # Load outcomes
    with open(OUTCOMES_FILE, 'r') as f:
        outcomes = json.load(f)
    outcomes_df = pd.DataFrame(list(outcomes.items()), columns=['PATIENT_ID', 'outcome'])


    outcomes_df['PATIENT_ID'] = outcomes_df['PATIENT_ID'].astype(int)
    
    datasets = {
        "Train": TRAIN_FILE,
        "Validation": VALIDATION_FILE,
        "Test": TEST_FILE
    }

    for name, file_path in datasets.items():
        if not file_path.exists():
            print(f"Dataset not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            patient_ids = json.load(f)
        patient_ids = [int(pid) for pid in patient_ids]

        # Filter outcomes for the current dataset
        dataset_outcomes = outcomes_df[outcomes_df['PATIENT_ID'].isin(patient_ids)]

        print(f"\n--- {name} Set ---")
        if 'outcome' in dataset_outcomes.columns:
            distribution = dataset_outcomes['outcome'].value_counts()
            percentage = dataset_outcomes['outcome'].value_counts(normalize=True) * 100
            
            print("Outcome Distribution (Count):")
            print(distribution)
            print("\nOutcome Distribution (Percentage):")
            print(percentage)
        else:
            print("'outcome' column not found.")

if __name__ == "__main__":
    inspect_split_distribution()

