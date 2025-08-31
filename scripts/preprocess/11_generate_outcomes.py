

# -*- coding: utf-8 -*-
"""
Generates a true outcome variable (Y) for the prognostic model.

This script processes raw clinical data to determine if a "severe outcome"
occurred for each patient within a 14-day window from their admission time.

A severe outcome is defined as either:
1.  An ICU admission during the inpatient stay.
2.  A re-admission for the same patient within 14 days of the initial admission.

The script produces a JSON file mapping each inpatient_id to a binary outcome.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
BASE_DATA_PATH = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"

# Input files
FILE_PATHS = {
    "inpatient": BASE_DATA_PATH / "HIS住院.xlsx",
    "emr": BASE_DATA_PATH / "嘉和EMR数据.xlsx",
}
ANALYSIS_READY_FILE = PROCESSED_DATA_PATH / "analysis_ready_data.json"

# Output file
OUTPUT_FILE = PROCESSED_DATA_PATH / "outcomes.json"

# Parameters
WINDOW_DAYS = 14
ICU_KEYWORDS = ['ICU', '重症监护']

def generate_outcomes():
    """
    Main function to generate and save the outcome labels.
    """
    print("Starting outcome generation process...")

    # --- 1. Load Data ---
    print("Loading raw and processed data files...")
    try:
        inpatient_df = pd.read_excel(FILE_PATHS["inpatient"])
        emr_df = pd.read_excel(FILE_PATHS["emr"])
        with open(ANALYSIS_READY_FILE, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # Create a DataFrame from the analysis-ready JSON
    patient_df = pd.DataFrame(analysis_data)
    patient_df['t0_admission_time'] = pd.to_datetime(patient_df['t0_admission_time'])

    # --- 2. Identify ICU Stays ---
    print("Identifying ICU stays from EMR and inpatient data...")
    # Assumption: ICU status is mentioned in the '药品医嘱' (drug order) column of
    # the inpatient data, e.g., "转入ICU" or in the EMR text.
    inpatient_df.rename(columns={"住院号码": "inpatient_id"}, inplace=True)
    emr_df.rename(columns={"住院号": "inpatient_id"}, inplace=True)

    # Create a regex pattern to search for ICU keywords
    icu_pattern = '|'.join(ICU_KEYWORDS)
    
    # Search in EMR data (checking '主诉' and '现病史')
    emr_df['emr_text'] = emr_df['主诉'].fillna('') + ' ' + emr_df['现病史'].fillna('')
    icu_in_emr = emr_df[emr_df['emr_text'].str.contains(icu_pattern, case=False, na=False)]
    
    # Search in Inpatient data (checking '药品医嘱')
    icu_in_inpatient = inpatient_df[inpatient_df['药品医嘱'].str.contains(icu_pattern, case=False, na=False)]

    icu_patient_ids = set(icu_in_emr['inpatient_id'].astype(str).unique()) | \
                      set(icu_in_inpatient['inpatient_id'].astype(str).unique())
    
    print(f"  - Found {len(icu_patient_ids)} unique inpatient stays with ICU keywords.")

    # --- 3. Identify Re-admissions ---
    print("Identifying re-admissions within a 14-day window...")
    # Assumption: '姓名' (Name) can be used to uniquely identify a patient across different stays.
    # This is a strong assumption and may not hold in a real-world scenario.
    
    # We need patient name, inpatient_id, and admission time.
    # Let's get the name from the raw inpatient file.
    patient_info_df = inpatient_df[['姓名', 'inpatient_id', '医嘱开始时间']].copy()
    patient_info_df.dropna(subset=['姓名', 'inpatient_id', '医嘱开始时间'], inplace=True)
    patient_info_df['inpatient_id'] = patient_info_df['inpatient_id'].astype(str)

    # Get t0 for each stay, coercing errors to NaT (Not a Time)
    patient_info_df['医嘱开始时间'] = pd.to_datetime(patient_info_df['医嘱开始时间'], errors='coerce')
    patient_info_df.dropna(subset=['医嘱开始时间'], inplace=True) # Drop rows that failed to parse

    admission_times = patient_info_df.groupby('inpatient_id')['医嘱开始时间'].min().reset_index()
    admission_times.rename(columns={'医嘱开始时间': 't0'}, inplace=True)

    # Map admission times back to patient names
    full_admissions = pd.merge(
        patient_info_df[['姓名', 'inpatient_id']].drop_duplicates(),
        admission_times,
        on='inpatient_id'
    )
    full_admissions.sort_values(by=['姓名', 't0'], inplace=True)

    # Calculate time difference between consecutive admissions for the same patient
    full_admissions['time_diff_days'] = full_admissions.groupby('姓名')['t0'].diff().dt.days
    
    # Identify stays that are re-admissions
    readmission_ids = set(
        full_admissions[full_admissions['time_diff_days'] <= WINDOW_DAYS]['inpatient_id']
    )
    print(f"  - Found {len(readmission_ids)} unique inpatient stays that were re-admissions.")

    # --- 4. Combine and Generate Final Outcome ---
    print("Combining criteria and generating final outcome labels...")
    severe_outcome_ids = icu_patient_ids | readmission_ids
    
    outcomes = {
        row['inpatient_id']: 1 if row['inpatient_id'] in severe_outcome_ids else 0
        for _, row in patient_df.iterrows()
    }

    positive_outcomes = sum(outcomes.values())
    print(f"  - Total patients with severe outcome: {positive_outcomes} ({positive_outcomes/len(outcomes):.2%})")

    # --- 5. Save the Outcomes ---
    print(f"Saving outcome data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(outcomes, f, indent=2)
    
    print("  - Done.")


if __name__ == "__main__":
    generate_outcomes()
