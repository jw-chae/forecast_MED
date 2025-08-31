
# -*- coding: utf-8 -*-
"""
Comprehensive preprocessing script for multimodal clinical data.

This script performs the end-to-end preprocessing of five raw data files
(HIS住院, LIS, PACS, EMR, HIS门诊) into a single, analysis-ready dataset.
Each row of the final dataset represents one inpatient stay, containing all
necessary features extracted from the first 72 hours of admission.
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, Any, List

# --- Configuration ---
# Define file paths for the input data.
# In a real pipeline, these would likely be passed as arguments or read from a config file.
BASE_DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "processed_data"

FILE_PATHS = {
    "inpatient": BASE_DATA_PATH / "HIS住院.xlsx",
    "lab": BASE_DATA_PATH / "LIS 去除身份证.xlsx",
    "pacs": BASE_DATA_PATH / "PACS影像数据 去身份证.xlsx",
    "emr": BASE_DATA_PATH / "嘉和EMR数据.xlsx",
    "outpatient": BASE_DATA_PATH / "HIS门诊.xlsx",
}

# --- Placeholder Data & Functions ---

# Placeholder for mapping drug names to ATC codes.
# In a real scenario, this would be a comprehensive dictionary or a database lookup.
def get_atc_code(drug_name: str) -> str:
    """
    Placeholder function to map a drug name to its primary ATC code.
    Returns a dummy code for demonstration.
    """
    if "胺" in drug_name:
        return "A11"  # Vitamins
    if "B" in drug_name:
        return "B03"  # Anti-anemic
    if "J" in drug_name:
        return "J01"
    return "N02"  # Analgesics


# Pre-computed dictionaries for normal ranges and critical thresholds for lab values.
# These would be derived from clinical guidelines or historical data analysis.
LAB_NORMALS = {
    'WBC': {'mean': 7.5, 'std': 2.5},
    'CRP': {'mean': 5.0, 'std': 3.0},
    'HGB': {'mean': 140, 'std': 20},
    # Add other common lab tests here
}

LAB_CRITICAL = {
    'WBC': {'low': 2.0, 'high': 20.0},
    'CRP': {'low': 0, 'high': 10.0},
    'HGB': {'low': 70, 'high': 180},
    # Add other common lab tests here
}

# --- Core Processing Functions ---

def load_data() -> Dict[str, pd.DataFrame]:
    """
    Loads all raw data from Excel files into pandas DataFrames.
    In a real pipeline, we would ideally convert these to a more efficient format
    like Parquet first to speed up loading.
    """
    print("Loading data from Excel files...")
    dataframes = {}
    for key, path in FILE_PATHS.items():
        try:
            dataframes[key] = pd.read_excel(path)
            print(f"  - Successfully loaded {key}: {path.name}")
        except FileNotFoundError:
            print(f"  - WARNING: File not found for {key} at {path}. Skipping.")
            dataframes[key] = pd.DataFrame() # Return empty dataframe if file not found
    return dataframes

def standardize_data(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Performs core data cleaning and standardization tasks.
    - Standardizes patient ID column names to 'inpatient_id'.
    - Converts date/time columns to datetime objects.
    - Cleans numeric columns.
    """
    print("\nStandardizing data...")

    # 1. Standardize Patient IDs
    id_map = {
        "inpatient": "住院号码",
        "lab": "INPATIENT_ID",
        "pacs": "门诊住院号",
        "emr": "住院号",
    }
    for key, df in dataframes.items():
        if key in id_map and id_map[key] in df.columns:
            df.rename(columns={id_map[key]: "inpatient_id"}, inplace=True)
            # Ensure inpatient_id is a string to avoid type mismatches during merge
            df['inpatient_id'] = df['inpatient_id'].astype(str)
            print(f"  - Standardized patient ID in {key}")

    # 2. Standardize Date/Time Columns
    date_cols = {
        "inpatient": "医嘱开始时间",
        "lab": "INSPECTION_DATE",
        "pacs": "检查日期",
        "outpatient": "诊断时间",
    }
    for key, col in date_cols.items():
        if col in dataframes[key].columns:
            dataframes[key][col] = pd.to_datetime(dataframes[key][col], errors='coerce')
            print(f"  - Standardized date column '{col}' in {key}")

    # 3. Clean Drug Dosage Column
    if "药品剂量" in dataframes["inpatient"].columns:
        df = dataframes["inpatient"]
        # Convert dosage to a numeric value in mg
        df["dosage_mg"] = df["药品剂量"].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        # Handle cases where original value was e.g., '.25mg'
        df.loc[df["药品剂量"].astype(str).str.startswith('.'), "dosage_mg"] /= 100
        dataframes["inpatient"] = df
        print("  - Cleaned drug dosage column in 'inpatient' data.")

    return dataframes


def process_inpatient_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes inpatient medication data to extract admission time (t0)
    and medications administered within the first 72 hours.
    """
    print("\nProcessing inpatient (HIS住院) data...")
    if df.empty or 'inpatient_id' not in df.columns:
        print("  - Inpatient data is empty or missing 'inpatient_id'. Skipping.")
        return pd.DataFrame()

    # Drop rows with missing time or patient ID
    df.dropna(subset=["医嘱开始时间", "inpatient_id"], inplace=True)

    # Determine admission time (t0) for each patient
    df["t0_admission_time"] = df.groupby("inpatient_id")["医嘱开始时间"].transform("min")

    # Filter records to the first 72 hours
    time_window = pd.Timedelta(hours=72)
    df_72h = df[df["医嘱开始时间"] <= df["t0_admission_time"] + time_window].copy()
    print(f"  - Filtered {len(df_72h)} records within the 72-hour window.")

    # Feature Engineering: Get unique ATC codes for each patient
    df_72h["atc_code"] = df_72h["药品医嘱"].apply(get_atc_code)
    
    patient_meds = df_72h.groupby("inpatient_id").agg(
        t0_admission_time=("t0_admission_time", "first"),
        meds_atc_list=("atc_code", lambda x: list(x.unique()))
    ).reset_index()
    
    print("  - Engineered medication features (ATC codes).")
    return patient_meds


def process_lab_data(df: pd.DataFrame, patient_t0: pd.Series) -> pd.DataFrame:
    """
    Processes lab (LIS) data to extract key features for each patient
    within the 72-hour window.
    """
    print("\nProcessing lab (LIS) data...")
    if df.empty or 'inpatient_id' not in df.columns:
        print("  - Lab data is empty or missing 'inpatient_id'. Skipping.")
        return pd.DataFrame()

    # Merge with t0 times to filter by patient-specific window
    df = df.merge(patient_t0.rename("t0_admission_time"), on="inpatient_id")
    
    # Filter records to the first 72 hours
    time_window = pd.Timedelta(hours=72)
    df_72h = df[df["INSPECTION_DATE"] <= df["t0_admission_time"] + time_window].copy()
    print(f"  - Filtered {len(df_72h)} lab records within the 72-hour window.")

    # Clean quantitative result
    df_72h["result_numeric"] = pd.to_numeric(df_72h["QUANTITATIVE_RESULT"], errors='coerce')
    df_72h.dropna(subset=["result_numeric", "CHINESE_NAME"], inplace=True)

    # Feature Engineering: Z-score, slope, critical flag
    def calculate_lab_features(group):
        features = {}
        for test_name, test_group in group.groupby("CHINESE_NAME"):
            # If we don't have predefined normal ranges, compute on-the-fly
            if test_name not in LAB_NORMALS:
                mean = test_group["result_numeric"].mean()
                std = test_group["result_numeric"].std() or 1.0  # avoid div/0
            else:
                mean = LAB_NORMALS[test_name]['mean']
                std = LAB_NORMALS[test_name]['std']

            # Sort by date to calculate slope correctly
            test_group = test_group.sort_values("INSPECTION_DATE")
            
            # Z-score of the first measurement
            first_val = test_group["result_numeric"].iloc[0]
            z_score = (first_val - mean) / std

            # 24-hour slope
            slope = 0.0
            if len(test_group) > 1:
                first_time = test_group["INSPECTION_DATE"].iloc[0]
                day_later = test_group[test_group["INSPECTION_DATE"] <= first_time + pd.Timedelta(hours=24)]
                if len(day_later) > 1:
                    last_val_24h = day_later["result_numeric"].iloc[-1]
                    slope = last_val_24h - first_val

            # Critical flag
            if test_name in LAB_CRITICAL:
                crit_low = LAB_CRITICAL[test_name]['low']
                crit_high = LAB_CRITICAL[test_name]['high']
                flag = 1 if not (crit_low <= first_val <= crit_high) else 0
            else:
                flag = 0  # Unknown critical range

            features[test_name] = f"{z_score:.1f}|{slope:.1f}|{flag}"
        return features

    lab_features = df_72h.groupby("inpatient_id").apply(calculate_lab_features)
    lab_features_df = lab_features.to_frame("lab_features_packed").reset_index()
    
    print("  - Engineered lab features (Z-score, slope, critical flag).")
    return lab_features_df


def clean_text(text: str) -> str:
    """A simple text cleaning function."""
    text = str(text)
    # Remove common boilerplate phrases
    boilerplate = ["请结合临床", "建议复查", "等可能", "考虑"]
    for phrase in boilerplate:
        text = text.replace(phrase, "")
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_text(text: str, max_length: int) -> str:
    """
    Placeholder function to simulate a BERT summarizer by truncating.
    A real implementation would use a transformer model.
    """
    return text[:max_length]

def process_text_data(pacs_df: pd.DataFrame, emr_df: pd.DataFrame, patient_t0: pd.Series) -> pd.DataFrame:
    """
    Processes PACS and EMR text data.
    """
    print("\nProcessing text (PACS & EMR) data...")
    all_patient_ids = patient_t0.index
    
    # Process PACS
    pacs_summary = pd.Series(index=all_patient_ids, dtype=str, name="pacs_summary")
    if not pacs_df.empty and 'inpatient_id' in pacs_df.columns:
        pacs_df = pacs_df.merge(patient_t0.rename("t0_admission_time"), on="inpatient_id")
        time_window = pd.Timedelta(hours=72)
        pacs_72h = pacs_df[pacs_df["检查日期"] <= pacs_df["t0_admission_time"] + time_window].copy()
        
        # Concatenate and clean text fields
        pacs_72h["full_report"] = pacs_72h["检查结论"].fillna('') + " " + pacs_72h["检查表现"].fillna('')
        pacs_72h["cleaned_report"] = pacs_72h["full_report"].apply(clean_text)
        
        # Group by patient and summarize
        pacs_grouped = pacs_72h.groupby("inpatient_id")["cleaned_report"].apply(" ".join)
        pacs_summary = pacs_grouped.apply(summarize_text, max_length=256)
        pacs_summary.name = "pacs_summary"
        print("  - Processed and summarized PACS data.")

    # Process EMR
    emr_history = pd.Series(index=all_patient_ids, dtype=str, name="emr_history")
    if not emr_df.empty and 'inpatient_id' in emr_df.columns:
        # Group by patient ID and join the history text, then truncate
        emr_grouped = emr_df.groupby("inpatient_id")["既往史"].apply(lambda x: " ".join(x.astype(str)))
        emr_history = emr_grouped.apply(lambda x: x[:512])
        emr_history.name = "emr_history"
        print("  - Processed and truncated EMR data.")

    # Combine into a single DataFrame
    text_features_df = pd.concat([pacs_summary, emr_history], axis=1).reset_index().rename(columns={'index': 'inpatient_id'})
    return text_features_df


def assemble_final_dataset(
    inpatient_features: pd.DataFrame,
    lab_features: pd.DataFrame,
    text_features: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Merges all processed DataFrames into a final, unified dataset.
    """
    print("\nAssembling final dataset...")
    
    # Start with the primary inpatient data
    final_df = inpatient_features
    
    # Merge lab features
    if not lab_features.empty:
        final_df = pd.merge(final_df, lab_features, on="inpatient_id", how="left")
    else:
        final_df['lab_features_packed'] = np.nan

    # Merge text features
    if not text_features.empty:
        final_df = pd.merge(final_df, text_features, on="inpatient_id", how="left")
    else:
        final_df['pacs_summary'] = ''
        final_df['emr_history'] = ''

    # Fill NaNs for list/dict/str columns to ensure consistent output format
    final_df['meds_atc_list'] = final_df['meds_atc_list'].apply(lambda x: x if isinstance(x, list) else [])
    final_df['lab_features_packed'] = final_df['lab_features_packed'].apply(lambda x: x if isinstance(x, dict) else {})
    final_df['pacs_summary'] = final_df['pacs_summary'].fillna('')
    final_df['emr_history'] = final_df['emr_history'].fillna('')

    # Convert t0 to string for JSON serialization
    final_df['t0_admission_time'] = final_df['t0_admission_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Add dummy demographics as per the requested output format
    # In a real scenario, this would come from the HIS住院 or HIS门诊 file
    final_df['demographics'] = final_df.apply(lambda _: {"age": np.random.randint(30, 80), "sex": np.random.choice(["男", "女"])}, axis=1)

    print(f"  - Final dataset assembled with {len(final_df)} patient records.")
    
    # Convert DataFrame to list of dictionaries
    return final_df.to_dict(orient='records')


def preprocess_data():
    """Main function to orchestrate the entire preprocessing pipeline."""
    
    # 1. Setup & Loading
    dataframes = load_data()
    
    # 2. Core Data Cleaning & Standardization
    dataframes = standardize_data(dataframes)
    
    # 3. Inpatient (HIS) Processing
    inpatient_features = process_inpatient_data(dataframes.get("inpatient", pd.DataFrame()))
    if inpatient_features.empty:
        print("Critical error: No inpatient data to form the base dataset. Exiting.")
        return

    # Create a series of patient IDs and their t0 for other functions to use
    patient_t0 = inpatient_features.set_index("inpatient_id")["t0_admission_time"]

    # 4. LIS (Lab) Processing
    lab_features = process_lab_data(dataframes.get("lab", pd.DataFrame()), patient_t0)
    
    # 5. PACS & EMR (Text) Processing
    text_features = process_text_data(
        dataframes.get("pacs", pd.DataFrame()),
        dataframes.get("emr", pd.DataFrame()),
        patient_t0
    )
    
    # 6. Final Assembly
    final_patient_data = assemble_final_dataset(
        inpatient_features,
        lab_features,
        text_features
    )
    
    # 7. Saving Output
    output_file = OUTPUT_PATH / "analysis_ready_data.json"
    print(f"\nSaving final data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_patient_data, f, ensure_ascii=False, indent=2)
    print("  - Done.")


if __name__ == "__main__":
    preprocess_data()
