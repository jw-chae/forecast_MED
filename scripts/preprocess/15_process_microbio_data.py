import pandas as pd
import os
import re

def process_microbiology_data(file_path):
    """
    미생물 검사 데이터를 전처리하여 분석용 데이터프레임들을 생성합니다.

    이 함수는 다음 단계를 수행합니다:
    1. INSPECTION_DATE를 datetime으로 변환합니다.
    2. 개인정보(PATIENT_NAME, ID_card) 컬럼을 삭제합니다.
    3. CHINESE_NAME을 분석하여 BUG와 RESULT 컬럼을 생성합니다.
       - "未培养到X" (X가 배양되지 않음) 형태는 음성(0)으로 처리합니다.
       - 그 외는 양성(1)으로 처리합니다.
    4. 환자-날짜별로 미생물 검사 결과를 피봇팅합니다.
    5. 날짜별 고유 입원 환자 수를 계산합니다.
    6. 임상 진단명을 기반으로 특정 전염병의 주간 발생 건수를 집계합니다.

    Args:
        file_path (str): 전처리할 원본 데이터 파일 경로 (Excel).

    Returns:
        dict: 'pivot', 'daily', 'epi' 세 개의 데이터프레임이 포함된 딕셔너리.
              데이터 로드 실패 시 None을 반환합니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다 - {file_path}")
        return None

    # 1. 데이터 로드 및 기본 전처리
    df = pd.read_excel(file_path)

    # 'ID_card' 컬럼이 존재할 경우 삭제
    if 'ID_card' in df.columns:
        df = df.drop(columns=['ID_card'])
    if 'PATIENT_NAME' in df.columns:
        df = df.drop(columns=['PATIENT_NAME'])

    # 2. 날짜 변환
    df['INSPECTION_DATE'] = pd.to_datetime(df['INSPECTION_DATE'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['INSPECTION_DATE'])
    
    # 3. BUG 및 RESULT 컬럼 생성
    def extract_bug_and_result(name):
        if pd.isna(name):
            return None, -1
        match = re.search(r'未培养到(.+)', str(name))
        if match:
            return match.group(1).strip(), 0  # 음성
        else:
            return str(name), 1  # 양성

    df[['BUG', 'RESULT']] = df['CHINESE_NAME'].apply(lambda x: pd.Series(extract_bug_and_result(x)))
    df = df[df['RESULT'] != -1] # 파싱 실패한 데이터 제외

    # 4. 환자-날짜별 피봇 (df_pivot)
    # INPATIENT_ID가 없는 경우를 대비하여 OUTPATIENT_ID와 결합하여 고유 식별자 생성
    df['PATIENT_ID'] = df['INPATIENT_ID'].fillna(df['OUTPATIENT_ID']).astype(str)
    
    df_pivot = df.pivot_table(
        index=['PATIENT_ID', 'INSPECTION_DATE'],
        columns='BUG',
        values='RESULT',
        fill_value=-1  # 검사 안함: -1, 음성: 0, 양성: 1
    ).reset_index()

    # 5. 날짜별 환자 수 (df_daily)
    df_daily = df[df['INPATIENT_ID'].notna()].groupby('INSPECTION_DATE')['INPATIENT_ID'].nunique().reset_index()
    df_daily.columns = ['date', 'unique_patients']

    # 6. 우선순위가 적용된 키워드 기반 공식 전염병 매핑 및 주간 집계 (df_epi)
    DISEASE_KEYWORDS_ORDERED = [
        # 상세 질병을 일반 질병보다 먼저 배치하여 우선순위 부여
        ("甲型肝炎", ["甲型肝炎"]),
        ("乙型肝炎", ["乙型肝炎"]),
        ("丙型肝炎", ["丙型肝炎"]),
        ("丁型肝炎", ["丁型肝炎"]),
        ("戊型肝炎", ["戊型肝炎"]),
        ("病毒性肝炎", ["肝炎"]), # '...肝炎'을 마지막에 배치

        ("人禽流感", ["人禽流感"]),
        ("人感染H7N9禽流感", ["H7N9"]),
        ("流行性感冒", ["流感"]), # '流感' 키워드로 '流感样病例' 포함

        ("新型冠状病毒肺炎", ["冠状病毒", "新冠"]),
        ("传染性非典", ["非典", "SARS"]),
        
        ("细菌性和阿米巴性痢疾", ["痢疾"]),
        ("其他感染性腹泻病", ["腹泻病"]),

        # 나머지 질병 목록 (가나다 순)
        ("艾滋病", ["艾滋病", "AIDS"]),
        ("包虫病", ["包虫病"]),
        ("斑疹伤寒", ["斑疹伤寒"]),
        ("白喉", ["白喉"]),
        ("百日咳", ["百日咳"]),
        ("布病", ["布病"]),
        ("炭疽", ["炭疽"]),
        ("登革热", ["登革热"]),
        ("出血热", ["出血热"]),
        ("风疹", ["风疹"]),
        ("钩体病", ["钩体病"]),
        ("黑热病", ["黑热病"]),
        ("猴痘", ["猴痘"]),
        ("霍乱", ["霍乱"]),
        ("急性出血性结膜炎", ["结膜炎"]),
        ("脊灰", ["脊灰"]),
        ("狂犬病", ["狂犬病"]),
        ("淋病", ["淋病"]),
        ("流行性腮腺炎", ["腮腺炎"]),
        ("流脑", ["流脑"]),
        ("麻风病", ["麻风病"]),
        ("麻疹", ["麻疹"]),
        ("梅毒", ["梅毒"]),
        ("疟疾", ["疟疾"]),
        ("肺结核", ["结核"]),
        ("伤寒和副伤寒", ["伤寒"]),
        ("手足口病", ["手足口"]),
        ("鼠疫", ["鼠疫"]),
        ("丝虫病", ["丝虫病"]),
        ("血吸虫病", ["血吸虫病"]),
        ("新生儿破伤风", ["破伤风"]),
        ("猩红热", ["猩红热"]),
        ("乙脑", ["乙脑"]),
    ]

    # 간단 정규화: 공백 제거, 대소문자 통일
    if 'CLINICAL_DIAGNOSES' in df.columns:
        df['CLINICAL_DIAGNOSES'] = df['CLINICAL_DIAGNOSES'].astype(str).str.strip().str.replace(r"\s+", "", regex=True)

    def map_diagnosis_by_keyword_ordered(diagnosis):
        if pd.isna(diagnosis):
            return 'Other'
        diag_str = str(diagnosis)
        for disease_name, keywords in DISEASE_KEYWORDS_ORDERED:
            for keyword in keywords:
                if keyword in diag_str:
                    return disease_name
        return 'Other'

    df['epi_category'] = df['CLINICAL_DIAGNOSES'].apply(map_diagnosis_by_keyword_ordered)
    
    # 주 단위로 리샘플링하여 카테고리별 환자 수 집계 (CLINICAL_DIAGNOSES 기반)
    df_epi = df.set_index('INSPECTION_DATE').groupby([pd.Grouper(freq='W-MON'), 'epi_category'])['PATIENT_ID'].nunique().unstack(fill_value=0)
    df_epi.columns.name = None
    df_epi = df_epi.reset_index()

    # --- LIS: 진단(epi_category) 기준 Top20 선정 (전체기간 고유 환자 수) ---
    try:
        total_unique_by_epi = (
            df.groupby('epi_category')['PATIENT_ID'].nunique().sort_values(ascending=False)
        )
        # 'Other'는 Top20 선정에서 제외
        epi_index_wo_other = [k for k in total_unique_by_epi.index if k != 'Other']
        top20_epi = list(total_unique_by_epi.loc[epi_index_wo_other].head(20).index)
    except Exception:
        top20_epi = []

    # 주간 × 진단(epi_category) 고유 환자 수 Top20 서브셋
    if top20_epi:
        for cat in top20_epi:
            if cat not in df_epi.columns:
                df_epi[cat] = 0
        df_weekly_epi_top20_unique = df_epi[['INSPECTION_DATE'] + top20_epi].copy()
    else:
        df_weekly_epi_top20_unique = df_epi[['INSPECTION_DATE']].copy()

    # 동일 Top20 진단에 대해 주간 방문 건수(레코드 수)도 산출
    df_weekly_epi_visits = df.set_index('INSPECTION_DATE').groupby([
        pd.Grouper(freq='W-MON'), 'epi_category'
    ])['PATIENT_ID'].size().unstack(fill_value=0).reset_index()
    if top20_epi:
        for cat in top20_epi:
            if cat not in df_weekly_epi_visits.columns:
                df_weekly_epi_visits[cat] = 0
        df_weekly_epi_top20_visits = df_weekly_epi_visits[['INSPECTION_DATE'] + top20_epi].copy()
    else:
        df_weekly_epi_top20_visits = df_weekly_epi_visits[['INSPECTION_DATE']].copy()

    # --- 비법정 진단 상위 4개(전체기간 고유 환자 수 기준)를 Others에서 분리하여 별도 컬럼으로 추가 ---
    # 진단 원문 정규화 컬럼
    df['diag_norm'] = df.get('CLINICAL_DIAGNOSES', pd.Series(index=df.index, dtype=str)).astype(str).str.strip()
    df['diag_norm'] = df['diag_norm'].str.replace(r"\s+", "", regex=True)
    # epi_category == 'Other' 중에서 상위 4개 진단(고유 환자 수 기준)
    try:
        other_mask = (df['epi_category'] == 'Other')
        top4_non_epi = (
            df.loc[other_mask].groupby('diag_norm')['PATIENT_ID'].nunique().sort_values(ascending=False).head(4).index.tolist()
        )
    except Exception:
        top4_non_epi = []

    # 주간 × top4_non_epi 고유 환자 수 피벗
    if top4_non_epi:
        df_weekly_top4_unique = df.set_index('INSPECTION_DATE').groupby([
            pd.Grouper(freq='W-MON'), 'diag_norm'
        ])['PATIENT_ID'].nunique().unstack(fill_value=0).reset_index()
        # 주간 × top4_non_epi 방문 건수 피벗
        df_weekly_top4_visits = df.set_index('INSPECTION_DATE').groupby([
            pd.Grouper(freq='W-MON'), 'diag_norm'
        ])['PATIENT_ID'].size().unstack(fill_value=0).reset_index()
        # df_epi(full) 사본에 병합 및 Others 조정
        df_epi_plus4 = df_epi.copy()
        for diag in top4_non_epi:
            if diag in df_weekly_top4_unique.columns:
                df_epi_plus4 = df_epi_plus4.merge(
                    df_weekly_top4_unique[['INSPECTION_DATE', diag]], on='INSPECTION_DATE', how='left'
                )
                df_epi_plus4[diag] = df_epi_plus4[diag].fillna(0).astype(int)
        if 'Other' in df_epi_plus4.columns:
            # Others에서 top4 고유 환자 수를 제외
            subtract_sum = 0
            for diag in top4_non_epi:
                if diag in df_epi_plus4.columns:
                    subtract_sum = subtract_sum + df_epi_plus4[diag]
            df_epi_plus4['Other'] = (df_epi_plus4['Other'] - subtract_sum).clip(lower=0)
        # Top20+4 (unique) 표 구성
        df_weekly_epi_top20_plus4_unique = df_epi_plus4[['INSPECTION_DATE'] + top20_epi + top4_non_epi] if top20_epi else df_epi_plus4[['INSPECTION_DATE'] + top4_non_epi]

        # 방문 건수 버전 생성을 위해 df_weekly_epi_visits 사본 사용
        df_visits_plus4 = df_weekly_epi_visits.copy()
        for diag in top4_non_epi:
            if diag in df_weekly_top4_visits.columns:
                df_visits_plus4 = df_visits_plus4.merge(
                    df_weekly_top4_visits[['INSPECTION_DATE', diag]], on='INSPECTION_DATE', how='left'
                )
                df_visits_plus4[diag] = df_visits_plus4[diag].fillna(0).astype(int)
        # Top20+4 (visits) 표 구성
        df_weekly_epi_top20_plus4_visits = df_visits_plus4[['INSPECTION_DATE'] + top20_epi + top4_non_epi] if top20_epi else df_visits_plus4[['INSPECTION_DATE'] + top4_non_epi]
    else:
        df_weekly_epi_top20_plus4_unique = df_weekly_epi_top20_unique.copy()
        df_weekly_epi_top20_plus4_visits = df_weekly_epi_top20_visits.copy()

    # --- LIS: 검사항목(Chinese Name) 기준 Top20 선정 (전체기간 고유 환자 수) ---
    try:
        total_unique_by_test = (
            df.groupby('CHINESE_NAME')['PATIENT_ID'].nunique().sort_values(ascending=False)
        )
        top20_tests = total_unique_by_test.head(20).index.tolist()
    except Exception:
        top20_tests = []

    # 주간 × 검사항목 고유 환자 수 피벗
    df_weekly_test_unique = df.set_index('INSPECTION_DATE').groupby([
        pd.Grouper(freq='W-MON'), 'CHINESE_NAME'
    ])['PATIENT_ID'].nunique().unstack(fill_value=0).reset_index()
    if top20_tests:
        for t in top20_tests:
            if t not in df_weekly_test_unique.columns:
                df_weekly_test_unique[t] = 0
        df_weekly_top20_tests_unique = df_weekly_test_unique[['INSPECTION_DATE'] + top20_tests].copy()
    else:
        df_weekly_top20_tests_unique = df_weekly_test_unique[['INSPECTION_DATE']].copy()

    # 동일 Top20 검사항목에 대해 주간 방문 건수(검사 시행 건수)도 집계
    df_weekly_test_visits = df.set_index('INSPECTION_DATE').groupby([
        pd.Grouper(freq='W-MON'), 'CHINESE_NAME'
    ])['PATIENT_ID'].size().unstack(fill_value=0).reset_index()
    if top20_tests:
        for t in top20_tests:
            if t not in df_weekly_test_visits.columns:
                df_weekly_test_visits[t] = 0
        df_weekly_top20_tests_visits = df_weekly_test_visits[['INSPECTION_DATE'] + top20_tests].copy()
    else:
        df_weekly_top20_tests_visits = df_weekly_test_visits[['INSPECTION_DATE']].copy()


    return {
        "pivot": df_pivot,
        "daily": df_daily,
        "epi": df_epi,
        "weekly_epi_top20_unique": df_weekly_epi_top20_unique,
        "weekly_epi_top20_visits": df_weekly_epi_top20_visits,
        "weekly_top20_tests_unique": df_weekly_top20_tests_unique,
        "weekly_top20_tests_visits": df_weekly_top20_tests_visits,
        "top20_epi": top20_epi,
        "top20_tests": top20_tests,
        "top4_non_epi": top4_non_epi,
        "weekly_epi_top20_plus4_unique": df_weekly_epi_top20_plus4_unique,
        "weekly_epi_top20_plus4_visits": df_weekly_epi_top20_plus4_visits,
    }

if __name__ == '__main__':
    # 데이터 파일 경로는 실제 환경에 맞게 수정해야 합니다.
    # 이 스크립트는 'scripts/preprocess' 에 위치하고, 데이터는 'data'에 위치한다고 가정합니다.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(base_dir, 'data', 'LIS 去除身份证.xlsx')

    results = process_microbiology_data(data_file)

    if results:
        print("### 전처리 결과 요약 ###")
        print("\n[1] 피봇 데이터프레임 (df_pivot) 샘플:")
        print(results['pivot'].head())
        
        print("\n[2] 일일 환자 수 데이터프레임 (df_daily) 샘플:")
        print(results['daily'].head())
        
        print("\n[3] 주간 전염병 통계 데이터프레임 (df_epi) 샘플:")
        print(results['epi'].head())
        
        # 결과 파일 저장
        output_dir = os.path.join(base_dir, 'processed_data')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results['pivot'].to_csv(os.path.join(output_dir, 'microbio_pivot.csv'), index=False, encoding='utf-8-sig')
        results['daily'].to_csv(os.path.join(output_dir, 'microbio_daily_counts.csv'), index=False, encoding='utf-8-sig')
        results['epi'].to_csv(os.path.join(output_dir, 'microbio_weekly_epi_counts_keyword.csv'), index=False, encoding='utf-8-sig')
        # 진단 기반 Top20 주간 통계 저장 (CLINICAL_DIAGNOSES 기반)
        results['weekly_epi_top20_unique'].to_csv(os.path.join(output_dir, 'lis_weekly_epi_top20_unique_counts.csv'), index=False, encoding='utf-8-sig')
        results['weekly_epi_top20_visits'].to_csv(os.path.join(output_dir, 'lis_weekly_epi_top20_visit_counts.csv'), index=False, encoding='utf-8-sig')
        # Top20 + 비법정 상위4 추가 버전 저장
        results['weekly_epi_top20_plus4_unique'].to_csv(os.path.join(output_dir, 'lis_weekly_epi_top20_plus4_unique_counts.csv'), index=False, encoding='utf-8-sig')
        results['weekly_epi_top20_plus4_visits'].to_csv(os.path.join(output_dir, 'lis_weekly_epi_top20_plus4_visit_counts.csv'), index=False, encoding='utf-8-sig')
        # Top20 검사항목 주간 통계 저장
        results['weekly_top20_tests_unique'].to_csv(os.path.join(output_dir, 'lis_weekly_top20_tests_unique_counts.csv'), index=False, encoding='utf-8-sig')
        results['weekly_top20_tests_visits'].to_csv(os.path.join(output_dir, 'lis_weekly_top20_tests_visit_counts.csv'), index=False, encoding='utf-8-sig')
        # Top20 목록도 참고용 저장
        pd.Series(results['top20_tests'], name='CHINESE_NAME').to_csv(os.path.join(output_dir, 'lis_top20_tests_list.csv'), index=False, encoding='utf-8-sig')
        print(f"\n결과가 '{output_dir}' 디렉토리에 저장되었습니다.")


