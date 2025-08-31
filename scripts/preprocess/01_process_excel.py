
import pandas as pd
import os
import json

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts', 'preprocess')

# List of Excel files to process
excel_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]

# Dictionary for translating column names (will be expanded)
column_translation = {
    # Common fields
    "姓名": "patient_name",
    "性别": "gender",
    "年龄": "age",
    "就诊卡号": "visit_card_id",
    "科室": "department",
    "诊断": "diagnosis",
    "主诉": "chief_complaint",
    "现病史": "history_of_present_illness",
    "既往史": "past_medical_history",
    "体格检查": "physical_examination",
    "检查项目": "examination_item",
    "检查结果": "examination_result",
    "危急值": "critical_value",
    "参考范围": "reference_range",
    "单位": "unit",
    "申请科室": "requesting_department",
    "报告时间": "report_time",
    "影像号": "imaging_id",
    "影像表现": "imaging_findings",
    "影像诊断": "imaging_diagnosis",
    "住院号": "hospitalization_id",
    "入院诊断": "admission_diagnosis",
    "出院诊断": "discharge_diagnosis",
}

def translate_columns(df, translation_dict):
    """Translates DataFrame columns based on a dictionary."""
    df = df.rename(columns=translation_dict)
    return df

def default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

import re

# ... (rest of the imports)

# ... (column_translation dictionary)

def extract_demographics(df):
    """
    Extracts age and gender from text columns using regex.
    """
    if 'physical_examination' in df.columns:
        # Extract age
        df['age'] = df['physical_examination'].str.extract(r'年龄[：:]?(\d+)')
        # Extract gender
        df['gender'] = df['physical_examination'].str.extract(r'性别[：:]?(男|女)')

    return df

def process_excel_file(file_path, output_dir):
    """
    Reads an Excel file, processes it, and saves as JSON.
    """
    try:
        df = pd.read_excel(file_path)
        
        # Translate column names
        df = translate_columns(df, column_translation)
        
        # Extract age and gender
        df = extract_demographics(df)

        # --- PII Removal ---
        if 'patient_name' in df.columns:
            df.drop(columns=['patient_name'], inplace=True)
        
        # Convert to list of dictionaries (after PII removal)
        records = df.to_dict('records')
        
        # Save processed data as JSON
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4, default=default_serializer)
            
        print(f"Successfully processed and saved {file_path} to {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    """
    Main function to orchestrate the processing of all Excel files.
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    for file_name in excel_files:
        file_path = os.path.join(DATA_DIR, file_name)
        process_excel_file(file_path, PROCESSED_DIR)

if __name__ == "__main__":
    main()
