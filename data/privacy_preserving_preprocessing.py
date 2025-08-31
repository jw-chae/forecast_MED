#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개인정보 보호 전처리 파이프라인
의료 데이터의 개인정보를 익명화하여 연구용 데이터셋으로 변환
"""

import pandas as pd
import numpy as np
import hashlib
import json
import re
from datetime import datetime
import os
from typing import Dict, List, Any, Tuple

class PrivacyPreservingPreprocessor:
    """개인정보 보호 전처리 클래스"""
    
    def __init__(self, output_dir: str = "anonymized_data"):
        self.output_dir = output_dir
        self.patient_id_mapping = {}  # 환자 ID 매핑 저장
        self.anonymization_log = []   # 익명화 로그
        
        # 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def identify_personal_info_columns(self) -> Dict[str, List[str]]:
        """개인정보 컬럼 패턴 정의"""
        return {
            'name': ['姓名', '患者姓名', 'PATIENT_NAME', '患儿家长姓名', 'CHINESE_NAME'],
            'id': ['住院号', '门诊号', '住院号码', 'OUTPATIENT_ID', 'INPATIENT_ID', '门诊住院号'],
            'phone': ['联系电话', 'phone', 'tel'],
            'address': ['户籍', '现住详细地址', '现住址', '户籍地址', 'address'],
            'birth': ['出生日期', 'birth', 'birthday'],
            'age': ['年龄', 'AGE_INPUT', 'age'],
            'gender': ['性别', 'PATIENT_SEX', 'gender'],
            'occupation': ['职业', 'occupation', 'job'],
            'workplace': ['患者工作单位', 'workplace', 'company']
        }
    
    def hash_name(self, name: str) -> str:
        """이름을 해시로 암호화"""
        if pd.isna(name) or name == '':
            return 'ANONYMOUS'
        return hashlib.sha256(str(name).encode('utf-8')).hexdigest()[:8]
    
    def mask_phone(self, phone: str) -> str:
        """전화번호 마스킹"""
        if pd.isna(phone) or phone == '':
            return 'MASKED'
        phone_str = str(phone)
        if len(phone_str) >= 7:
            return phone_str[:3] + '****' + phone_str[-4:]
        return 'MASKED'
    
    def generalize_address(self, address: str) -> str:
        """주소 일반화 (시/도 수준)"""
        if pd.isna(address) or address == '':
            return 'GENERALIZED'
        
        address_str = str(address)
        
        # 중국 주소 패턴 처리
        if '省' in address_str:
            province = address_str.split('省')[0] + '省'
            return province
        elif '市' in address_str:
            city = address_str.split('市')[0] + '市'
            return city
        elif '区' in address_str:
            district = address_str.split('区')[0] + '区'
            return district
        
        return 'GENERALIZED'
    
    def age_to_group(self, age: Any) -> str:
        """연령을 연령대 그룹으로 변환"""
        if pd.isna(age) or age == '':
            return 'UNKNOWN'
        
        try:
            age_num = int(str(age).replace('岁', ''))
            if age_num < 18:
                return '0-17'
            elif age_num < 30:
                return '18-29'
            elif age_num < 40:
                return '30-39'
            elif age_num < 50:
                return '40-49'
            elif age_num < 60:
                return '50-59'
            elif age_num < 70:
                return '60-69'
            else:
                return '70+'
        except:
            return 'UNKNOWN'
    
    def birth_to_age_group(self, birth_date: str) -> str:
        """생년월일을 연령대 그룹으로 변환"""
        if pd.isna(birth_date) or birth_date == '':
            return 'UNKNOWN'
        
        try:
            # 다양한 날짜 형식 처리
            if '-' in str(birth_date):
                year = int(str(birth_date).split('-')[0])
            else:
                year = int(str(birth_date)[:4])
            
            current_year = datetime.now().year
            age = current_year - year
            return self.age_to_group(age)
        except:
            return 'UNKNOWN'
    
    def generate_patient_id(self, original_id: str) -> str:
        """환자 ID를 의사난수 ID로 변환"""
        if pd.isna(original_id) or original_id == '':
            return 'P000000'
        
        if original_id not in self.patient_id_mapping:
            new_id = f"P{len(self.patient_id_mapping) + 1:06d}"
            self.patient_id_mapping[original_id] = new_id
        
        return self.patient_id_mapping[original_id]
    
    def anonymize_dataframe(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """데이터프레임 익명화"""
        df_anonymized = df.copy()
        personal_info_patterns = self.identify_personal_info_columns()
        
        # 각 컬럼에 대해 익명화 적용
        for col in df_anonymized.columns:
            col_lower = col.lower()
            
            # 이름 컬럼 처리
            if any(pattern.lower() in col_lower for pattern in personal_info_patterns['name']):
                df_anonymized[col] = df_anonymized[col].apply(self.hash_name)
                self.anonymization_log.append(f"{file_name}: {col} -> 해시화")
            
            # ID 컬럼 처리
            elif any(pattern.lower() in col_lower for pattern in personal_info_patterns['id']):
                df_anonymized[col] = df_anonymized[col].apply(self.generate_patient_id)
                self.anonymization_log.append(f"{file_name}: {col} -> 의사난수 ID")
            
            # 전화번호 컬럼 처리
            elif any(pattern.lower() in col_lower for pattern in personal_info_patterns['phone']):
                df_anonymized[col] = df_anonymized[col].apply(self.mask_phone)
                self.anonymization_log.append(f"{file_name}: {col} -> 마스킹")
            
            # 주소 컬럼 처리
            elif any(pattern.lower() in col_lower for pattern in personal_info_patterns['address']):
                df_anonymized[col] = df_anonymized[col].apply(self.generalize_address)
                self.anonymization_log.append(f"{file_name}: {col} -> 일반화")
            
            # 연령 컬럼 처리
            elif any(pattern.lower() in col_lower for pattern in personal_info_patterns['age']):
                df_anonymized[col] = df_anonymized[col].apply(self.age_to_group)
                self.anonymization_log.append(f"{file_name}: {col} -> 연령대 그룹")
            
            # 생년월일 컬럼 처리
            elif any(pattern.lower() in col_lower for pattern in personal_info_patterns['birth']):
                df_anonymized[col] = df_anonymized[col].apply(self.birth_to_age_group)
                self.anonymization_log.append(f"{file_name}: {col} -> 연령대 그룹")
        
        return df_anonymized
    
    def validate_anonymization(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """익명화 품질 검증"""
        validation_report = {
            'total_rows': len(original_df),
            'total_columns': len(original_df.columns),
            'anonymized_columns': [],
            'privacy_score': 0,
            'data_utility_score': 0
        }
        
        personal_info_patterns = self.identify_personal_info_columns()
        anonymized_count = 0
        
        for col in original_df.columns:
            col_lower = col.lower()
            
            # 개인정보 컬럼인지 확인
            is_personal_info = any(
                any(pattern.lower() in col_lower for pattern in patterns)
                for patterns in personal_info_patterns.values()
            )
            
            if is_personal_info:
                anonymized_count += 1
                validation_report['anonymized_columns'].append(col)
                
                # 익명화 품질 확인
                original_unique = original_df[col].nunique()
                anonymized_unique = anonymized_df[col].nunique()
                
                if original_unique > anonymized_unique:
                    validation_report['privacy_score'] += 1
        
        validation_report['privacy_score'] = (anonymized_count / len(original_df.columns)) * 100
        validation_report['data_utility_score'] = 100 - validation_report['privacy_score']
        
        return validation_report
    
    def process_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """단일 파일 처리"""
        print(f"처리 중: {file_path}")
        
        # 파일 로드
        if 'PACS' in file_path:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
        else:
            df = pd.read_excel(file_path)
        
        # 익명화 전 검증
        print(f"  원본 데이터: {df.shape[0]}행 x {df.shape[1]}열")
        
        # 익명화 처리
        file_name = os.path.basename(file_path)
        df_anonymized = self.anonymize_dataframe(df, file_name)
        
        # 익명화 품질 검증
        validation_report = self.validate_anonymization(df, df_anonymized)
        
        print(f"  익명화 완료: {len(validation_report['anonymized_columns'])}개 컬럼 처리")
        print(f"  개인정보 보호 점수: {validation_report['privacy_score']:.1f}%")
        
        return df_anonymized, validation_report
    
    def save_anonymized_data(self, data_dict: Dict[str, pd.DataFrame], validation_reports: Dict[str, Dict]):
        """익명화된 데이터 저장"""
        
        # 1. 익명화된 Excel 파일들 저장
        for file_name, df in data_dict.items():
            output_path = os.path.join(self.output_dir, f"anonymized_{file_name}")
            df.to_excel(output_path, index=False)
            print(f"저장됨: {output_path}")
        
        # 2. 환자 ID 매핑 저장
        mapping_path = os.path.join(self.output_dir, "patient_id_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.patient_id_mapping, f, ensure_ascii=False, indent=2)
        print(f"환자 ID 매핑 저장됨: {mapping_path}")
        
        # 3. 익명화 로그 저장
        log_path = os.path.join(self.output_dir, "anonymization_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.anonymization_log, f, ensure_ascii=False, indent=2)
        print(f"익명화 로그 저장됨: {log_path}")
        
        # 4. 검증 보고서 저장
        validation_path = os.path.join(self.output_dir, "validation_reports.json")
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_reports, f, ensure_ascii=False, indent=2)
        print(f"검증 보고서 저장됨: {validation_path}")
        
        # 5. 통합 JSON 데이터셋 생성
        unified_data = []
        for file_name, df in data_dict.items():
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['source_file'] = file_name
                unified_data.append(row_dict)
        
        unified_path = os.path.join(self.output_dir, "unified_anonymized_dataset.json")
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        print(f"통합 데이터셋 저장됨: {unified_path}")
        
        return {
            'excel_files': list(data_dict.keys()),
            'mapping_file': mapping_path,
            'log_file': log_path,
            'validation_file': validation_path,
            'unified_dataset': unified_path
        }

def main():
    """메인 실행 함수"""
    print("=== 개인정보 보호 전처리 파이프라인 시작 ===")
    
    # 전처리기 초기화
    preprocessor = PrivacyPreservingPreprocessor()
    
    # 처리할 파일 목록
    files_to_process = [
        '嘉和EMR数据.xlsx',
        'PACS影像数据 去身份证.xlsx',
        'LIS 去除身份证.xlsx',
        'HIS住院.xlsx',
        'HIS门诊.xlsx'
    ]
    
    # 각 파일 처리
    anonymized_data = {}
    validation_reports = {}
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            try:
                df_anonymized, validation_report = preprocessor.process_file(file_path)
                file_name = os.path.basename(file_path)
                anonymized_data[file_name] = df_anonymized
                validation_reports[file_name] = validation_report
            except Exception as e:
                print(f"오류 발생 ({file_path}): {e}")
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    # 결과 저장
    if anonymized_data:
        saved_files = preprocessor.save_anonymized_data(anonymized_data, validation_reports)
        
        print("\n=== 처리 완료 ===")
        print(f"총 {len(anonymized_data)}개 파일 처리됨")
        print(f"출력 디렉토리: {preprocessor.output_dir}")
        print(f"총 환자 ID 매핑: {len(preprocessor.patient_id_mapping)}개")
        print(f"익명화 로그: {len(preprocessor.anonymization_log)}개 항목")
        
        # 품질 요약
        avg_privacy_score = np.mean([report['privacy_score'] for report in validation_reports.values()])
        print(f"평균 개인정보 보호 점수: {avg_privacy_score:.1f}%")
        
    else:
        print("처리할 데이터가 없습니다.")

if __name__ == "__main__":
    main() 