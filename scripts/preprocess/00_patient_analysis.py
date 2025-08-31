#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환자 이름 추출 및 분석 스크립트
각 Excel 파일별로 고유 환자 수를 분석하고, 완전한 정보가 있는 환자들을 식별합니다.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PatientAnalyzer:
    def __init__(self, data_dir="../../data"):
        self.data_dir = Path(data_dir)
        self.results = {}
        # 각 파일별 환자 고유번호 컬럼 지정
        self.id_columns = {
            'emr': '住院号',
            'pacs': '住院号',
            'lis': 'PATIENT_ID',
            'his_inpatient': '住院号码',
            'his_outpatient': '门诊号',
        }
        
    def find_patient_id_column(self, df, file_type):
        """환자 고유번호 컬럼을 자동으로 찾습니다"""
        # 우선순위별 컬럼명 후보들
        candidates = {
            'emr': ['住院号', '患者ID', '患者编号', '病历号'],
            'pacs': ['住院号', '患者ID', '患者编号', '检查号'],
            'lis': ['PATIENT_ID', '患者ID', '患者编号', '检验号'],
            'his_inpatient': ['住院号码', '患者ID', '患者编号'],
            'his_outpatient': ['门诊号', '患者ID', '患者编号']
        }
        
        if file_type in candidates:
            for col in candidates[file_type]:
                if col in df.columns:
                    return col
        
        # ID가 포함된 컬럼 찾기
        id_cols = [col for col in df.columns if 'ID' in col or 'id' in col or '号' in col]
        if id_cols:
            return id_cols[0]
            
        return None

    def analyze_emr_data(self):
        print("=== EMR Data Analysis ===")
        file_path = self.data_dir / "嘉和EMR数据.xlsx"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        df = pd.read_excel(file_path)
        id_col = self.id_columns['emr']
        if id_col not in df.columns:
            print(f"Patient ID column not found: {id_col}")
            return None
        total_records = len(df)
        unique_patients = df[id_col].nunique()
        null_ids = df[id_col].isnull().sum()
        complete_info_cols = [id_col, '主诉', '现病史', '主诊断']
        complete_info_mask = df[complete_info_cols].notna().all(axis=1)
        complete_patients = df[complete_info_mask][id_col].nunique()
        diagnosis_counts = df['主诊断'].value_counts().head(10)
        results = {
            'file': 'EMR',
            'total_records': total_records,
            'unique_patients': unique_patients,
            'null_ids': null_ids,
            'complete_patients': complete_patients,
            'diagnosis_counts': diagnosis_counts.to_dict(),
            'sample_patient_ids': df[id_col].dropna().unique()[:5].tolist()
        }
        print(f"Total records: {total_records}")
        print(f"Unique patients (by ID): {unique_patients}")
        print(f"Records with null ID: {null_ids}")
        print(f"Patients with complete info: {complete_patients}")
        print(f"Sample patient IDs: {results['sample_patient_ids']}")
        return results

    def analyze_pacs_data(self):
        print("\n=== PACS Data Analysis ===")
        file_path = self.data_dir / "PACS影像数据 去身份证.xlsx"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        df = pd.read_excel(file_path)
        id_col = self.find_patient_id_column(df, 'pacs')
        if id_col is None:
            print(f"Patient ID column not found in PACS data")
            print(f"Available columns: {list(df.columns)}")
            return None
        print(f"Using patient ID column: {id_col}")
        total_records = len(df)
        unique_patients = df[id_col].nunique()
        null_ids = df[id_col].isnull().sum()
        complete_info_cols = [id_col, '检查结论', '检查表现']
        complete_info_mask = df[complete_info_cols].notna().all(axis=1)
        complete_patients = df[complete_info_mask][id_col].nunique()
        if '检查结论' in df.columns:
            exam_type_counts = df['检查结论'].value_counts().head(10)
        else:
            exam_type_counts = pd.Series()
        results = {
            'file': 'PACS',
            'total_records': total_records,
            'unique_patients': unique_patients,
            'null_ids': null_ids,
            'complete_patients': complete_patients,
            'exam_type_counts': exam_type_counts.to_dict(),
            'sample_patient_ids': df[id_col].dropna().unique()[:5].tolist()
        }
        print(f"Total records: {total_records}")
        print(f"Unique patients (by ID): {unique_patients}")
        print(f"Records with null ID: {null_ids}")
        print(f"Patients with complete info: {complete_patients}")
        print(f"Sample patient IDs: {results['sample_patient_ids']}")
        return results

    def analyze_lis_data(self):
        print("\n=== LIS Data Analysis ===")
        file_path = self.data_dir / "LIS 去除身份证.xlsx"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        df = pd.read_excel(file_path)
        id_col = self.find_patient_id_column(df, 'lis')
        if id_col is None:
            print(f"Patient ID column not found in LIS data")
            print(f"Available columns: {list(df.columns)}")
            return None
        print(f"Using patient ID column: {id_col}")
        total_records = len(df)
        unique_patients = df[id_col].nunique()
        null_ids = df[id_col].isnull().sum()
        complete_info_cols = [id_col, 'TEST_ORDER_NAME', 'QUANTITATIVE_RESULT']
        complete_info_mask = df[complete_info_cols].notna().all(axis=1)
        complete_patients = df[complete_info_mask][id_col].nunique()
        if 'TEST_ORDER_NAME' in df.columns:
            test_type_counts = df['TEST_ORDER_NAME'].value_counts().head(10)
        else:
            test_type_counts = pd.Series()
        results = {
            'file': 'LIS',
            'total_records': total_records,
            'unique_patients': unique_patients,
            'null_ids': null_ids,
            'complete_patients': complete_patients,
            'test_type_counts': test_type_counts.to_dict(),
            'sample_patient_ids': df[id_col].dropna().unique()[:5].tolist()
        }
        print(f"Total records: {total_records}")
        print(f"Unique patients (by ID): {unique_patients}")
        print(f"Records with null ID: {null_ids}")
        print(f"Patients with complete info: {complete_patients}")
        print(f"Sample patient IDs: {results['sample_patient_ids']}")
        return results

    def analyze_his_inpatient_data(self):
        print("\n=== HIS Inpatient Data Analysis ===")
        file_path = self.data_dir / "HIS住院.xlsx"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        df = pd.read_excel(file_path)
        id_col = self.id_columns['his_inpatient']
        if id_col not in df.columns:
            print(f"Patient ID column not found: {id_col}")
            return None
        total_records = len(df)
        unique_patients = df[id_col].nunique()
        null_ids = df[id_col].isnull().sum()
        complete_info_cols = [id_col, '药品医嘱']
        complete_info_mask = df[complete_info_cols].notna().all(axis=1)
        complete_patients = df[complete_info_mask][id_col].nunique()
        if '药品医嘱' in df.columns:
            medication_counts = df['药品医嘱'].value_counts().head(10)
        else:
            medication_counts = pd.Series()
        results = {
            'file': 'HIS_Inpatient',
            'total_records': total_records,
            'unique_patients': unique_patients,
            'null_ids': null_ids,
            'complete_patients': complete_patients,
            'medication_counts': medication_counts.to_dict(),
            'sample_patient_ids': df[id_col].dropna().unique()[:5].tolist()
        }
        print(f"Total records: {total_records}")
        print(f"Unique patients (by ID): {unique_patients}")
        print(f"Records with null ID: {null_ids}")
        print(f"Patients with complete info: {complete_patients}")
        print(f"Sample patient IDs: {results['sample_patient_ids']}")
        return results

    def analyze_his_outpatient_data(self):
        print("\n=== HIS Outpatient Data Analysis ===")
        file_path = self.data_dir / "HIS门诊.xlsx"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        df = pd.read_excel(file_path)
        id_col = self.id_columns['his_outpatient']
        if id_col not in df.columns:
            print(f"Patient ID column not found: {id_col}")
            return None
        total_records = len(df)
        unique_patients = df[id_col].nunique()
        null_ids = df[id_col].isnull().sum()
        complete_info_cols = [id_col, '主诉', '主诊断']
        complete_info_mask = df[complete_info_cols].notna().all(axis=1)
        complete_patients = df[complete_info_mask][id_col].nunique()
        if '就诊科室' in df.columns:
            department_counts = df['就诊科室'].value_counts().head(10)
        else:
            department_counts = pd.Series()
        results = {
            'file': 'HIS_Outpatient',
            'total_records': total_records,
            'unique_patients': unique_patients,
            'null_ids': null_ids,
            'complete_patients': complete_patients,
            'department_counts': department_counts.to_dict(),
            'sample_patient_ids': df[id_col].dropna().unique()[:5].tolist()
        }
        print(f"Total records: {total_records}")
        print(f"Unique patients (by ID): {unique_patients}")
        print(f"Records with null ID: {null_ids}")
        print(f"Patients with complete info: {complete_patients}")
        print(f"Sample patient IDs: {results['sample_patient_ids']}")
        return results

    def cross_analysis(self):
        print("\n=== Cross-file Patient ID Analysis ===")
        all_patients = {}
        files = [
            ("EMR", "嘉和EMR数据.xlsx", 'emr'),
            ("PACS", "PACS影像数据 去身份证.xlsx", 'pacs'),
            ("LIS", "LIS 去除身份证.xlsx", 'lis'),
            ("HIS_Inpatient", "HIS住院.xlsx", 'his_inpatient'),
            ("HIS_Outpatient", "HIS门诊.xlsx", 'his_outpatient')
        ]
        for file_type, filename, file_key in files:
            file_path = self.data_dir / filename
            if file_path.exists():
                df = pd.read_excel(file_path)
                id_col = self.find_patient_id_column(df, file_key)
                if id_col is not None:
                    patients = set(df[id_col].dropna().unique())
                    all_patients[file_type] = patients
                    print(f"{file_type}: {len(patients)} patients (by ID: {id_col})")
        if len(all_patients) >= 2:
            file_types = list(all_patients.keys())
            common_patients = set.intersection(*all_patients.values())
            print(f"\nPatients present in all files: {len(common_patients)}")
            patients_in_multiple_files = set()
            for i in range(len(file_types)):
                for j in range(i+1, len(file_types)):
                    intersection = all_patients[file_types[i]] & all_patients[file_types[j]]
                    patients_in_multiple_files.update(intersection)
            print(f"Patients present in at least 2 files: {len(patients_in_multiple_files)}")
            unique_by_file = {}
            for file_type in file_types:
                other_patients = set()
                for other_type, other_patients_set in all_patients.items():
                    if other_type != file_type:
                        other_patients.update(other_patients_set)
                unique_patients = all_patients[file_type] - other_patients
                unique_by_file[file_type] = unique_patients
                print(f"Patients only in {file_type}: {len(unique_patients)}")
        return all_patients

    def create_visualizations(self):
        print("\n=== Visualization ===")
        summary_data = []
        for key, result in self.results.items():
            if result and isinstance(result, dict) and 'file' in result:
                summary_data.append({
                    'File': result['file'],
                    'Total Records': result['total_records'],
                    'Unique Patients': result['unique_patients'],
                    'Complete Info Patients': result['complete_patients']
                })
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes[0, 0].bar(df_summary['File'], df_summary['Unique Patients'])
            axes[0, 0].set_title('Unique Patients per File')
            axes[0, 0].set_ylabel('Number of Patients')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 1].bar(df_summary['File'], df_summary['Total Records'])
            axes[0, 1].set_title('Total Records per File')
            axes[0, 1].set_ylabel('Number of Records')
            axes[0, 1].tick_params(axis='x', rotation=45)
            complete_ratio = df_summary['Complete Info Patients'] / df_summary['Unique Patients'] * 100
            axes[1, 0].bar(df_summary['File'], complete_ratio)
            axes[1, 0].set_title('Complete Info Patient Ratio (%)')
            axes[1, 0].set_ylabel('Ratio (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            avg_records = df_summary['Total Records'] / df_summary['Unique Patients']
            axes[1, 1].bar(df_summary['File'], avg_records)
            axes[1, 1].set_title('Average Records per Patient')
            axes[1, 1].set_ylabel('Avg. Records')
            axes[1, 1].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig('../../visualizations/patient_analysis_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def convert_to_json_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {k: self.convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return self.convert_to_json_serializable(obj.to_dict())
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj

    def save_results(self):
        """결과를 JSON 파일로 저장"""
        output_file = "../../processed_data/patient_analysis_results.json"
        
        # 결과 정리
        clean_results = {}
        for key, result in self.results.items():
            if result:
                clean_results[key] = self.convert_to_json_serializable(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_file}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("환자 이름 추출 및 분석을 시작합니다...")
        
        # 각 파일별 분석
        self.results['emr'] = self.analyze_emr_data()
        self.results['pacs'] = self.analyze_pacs_data()
        self.results['lis'] = self.analyze_lis_data()
        self.results['his_inpatient'] = self.analyze_his_inpatient_data()
        self.results['his_outpatient'] = self.analyze_his_outpatient_data()
        
        # 파일 간 교차 분석
        self.results['cross_analysis'] = self.cross_analysis()
        
        # 시각화 생성
        self.create_visualizations()
        
        # 결과 저장
        self.save_results()
        
        print("\n분석이 완료되었습니다!")

if __name__ == "__main__":
    analyzer = PatientAnalyzer()
    analyzer.run_analysis() 