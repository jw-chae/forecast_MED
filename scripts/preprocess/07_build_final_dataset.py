import pandas as pd
import os
import json
import re

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'final_unified_dataset.json')

def parse_dosage(dosage_str):
    """Parses a dosage string like '50mg' into value and unit."""
    if not isinstance(dosage_str, str):
        return None, None
    
    match = re.match(r'([\d\.]+)\s*([a-zA-Z\s/]+)', dosage_str)
    if match:
        return float(match.group(1)), match.group(2).strip()
    
    # Handle cases with no clear unit
    try:
        return float(dosage_str), None
    except (ValueError, TypeError):
        return None, None

def process_outpatient_data(filepath):
    """Processes the rich outpatient data file."""
    df = pd.read_excel(filepath)
    records = []
    for _, row in df.iterrows():
        # Combine text fields into a single narrative
        narrative = (
            f"Chief Complaint: {row.get('主诉', '')}\n"
            f"History of Present Illness: {row.get('现病史', '')}\n"
            f"Past Medical History: {row.get('既往史', '')}\n"
            f"Physical Examination: {row.get('体格检查', '')}"
        )
        
        dosage_val, dosage_unit = parse_dosage(row.get('药品剂量'))

        records.append({
            "source": "outpatient",
            "visit_id": row.get('门诊号'),
            "department": row.get('就诊科室'),
            "visit_time": row.get('诊断时间'),
            "clinical_narrative": narrative.strip(),
            "primary_diagnosis": row.get('主诊断'),
            "primary_diagnosis_icd": row.get('主诊断ICD编码'),
            "medications": [{
                "name": row.get('药品名称'),
                "dosage_value": dosage_val,
                "dosage_unit": dosage_unit,
                "quantity": row.get('药品数量')
            }]
        })
    return records

def process_imaging_data(filepath):
    """Processes the PACS imaging reports."""
    df = pd.read_excel(filepath)
    records = []
    for _, row in df.iterrows():
        records.append({
            "source": "imaging",
            "visit_id": row.get('门诊住院号'),
            "exam_id": row.get('检查号'),
            "exam_date": row.get('检查日期'),
            "reason_for_exam": row.get('疾病名称'),
            "findings_text": row.get('检查表现'),
            "conclusion_text": row.get('检查结论'),
            # Use the conclusion as the primary diagnosis from this source
            "primary_diagnosis": row.get('检查结论') 
        })
    return records

def process_emr_data(filepath):
    """Processes the inpatient EMR data."""
    df = pd.read_excel(filepath)
    records = []
    for _, row in df.iterrows():
        narrative = (
            f"Chief Complaint: {row.get('主诉', '')}\n"
            f"History of Present Illness: {row.get('现病史', '')}\n"
            f"Past Medical History: {row.get('既往史', '')}\n"
            f"Physical Examination: {row.get('体格检查', '')}\n"
            f"First Progress Note: {row.get('首次病程记录', '')}"
        )
        records.append({
            "source": "emr",
            "visit_id": row.get('住院号'),
            "admission_narrative": narrative.strip(),
            "discharge_summary": row.get('出院记录'),
            "primary_diagnosis": row.get('主诊断'),
            "primary_diagnosis_icd": row.get('主诊断编码')
        })
    return records


def main():
    """
    Main function to build the final, structured dataset based on the new plans.
    """
    all_records = []
    
    print("Processing Outpatient Data...")
    outpatient_path = os.path.join(DATA_DIR, 'HIS门诊.xlsx')
    all_records.extend(process_outpatient_data(outpatient_path))
    
    print("Processing Imaging Data...")
    imaging_path = os.path.join(DATA_DIR, 'PACS影像数据 去身份证.xlsx')
    all_records.extend(process_imaging_data(imaging_path))

    print("Processing EMR Data...")
    emr_path = os.path.join(DATA_DIR, '嘉和EMR数据.xlsx')
    all_records.extend(process_emr_data(emr_path))

    # Note: We are intentionally leaving out LIS and inpatient medication data for now
    # to focus on the diagnosis-rich text sources, as per the feedback.

    print(f"Saving final unified dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Use a custom serializer for pandas datetime objects if they appear
        def default_serializer(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if pd.isna(obj):
                return None
            raise TypeError(f"Type {type(obj)} not serializable")
        json.dump(all_records, f, ensure_ascii=False, indent=4, default=default_serializer)
        
    print(f"Processing complete. {len(all_records)} records saved.")

if __name__ == "__main__":
    main()
