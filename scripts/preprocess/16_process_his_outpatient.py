import pandas as pd
import os
import re

def process_his_outpatient_data(file_path):
    """
    HIS 외래(门诊) 데이터를 전처리하여 주간 전염병 통계와 추가 정보를 생성합니다.

    이 함수는 다음 단계를 수행합니다:
    1. 'HIS门诊.xlsx' 파일을 로드합니다.
    2. 총 고유 환자 수를 계산합니다.
    3. 상위 10개 진료과 통계를 집계합니다.
    4. '诊断时间'(진단 시간) 컬럼을 datetime 형식으로 변환합니다.
    5. '主诊断'(주 진단명)에 우선순위 기반 키워드 매핑을 적용하여 공식 전염병으로 분류합니다.
    6. '门诊号'(외래 번호)를 기준으로 주간별/질병별 고유 환자 수를 집계합니다.
    7. 방문 기록 건수 기준 주간 Top 20 질병 통계를 추가로 생성합니다.
    8. 결과에 모든 공식 질병 컬럼이 포함되도록 합니다.

    Args:
        file_path (str): 전처리할 원본 데이터 파일 경로 (Excel).

    Returns:
        tuple: (주간 통계 DF, 총 고유 환자 수, 상위 진료과 Series)
               데이터 로드 실패 시 (None, None, None)을 반환합니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다 - {file_path}")
        return None, None, None

    df = pd.read_excel(file_path)

    # --- 추가 통계 정보 집계 ---
    # 1. 총 고유 환자 수
    total_unique_patients = df['门诊号'].nunique()

    # 2. 상위 10개 진료과
    top_10_departments = df['就诊科室'].value_counts().nlargest(10)

    # 필요한 컬럼만 선택하여 이후 작업 수행
    df = df[['诊断时间', '主诊断', '门诊号']].copy()
    df.columns = ['diagnosis_time', 'primary_diagnosis', 'outpatient_id']

    # --- 데이터 정제 ---
    df['diagnosis_time'] = pd.to_datetime(df['diagnosis_time'], errors='coerce')
    df.dropna(subset=['diagnosis_time', 'primary_diagnosis', 'outpatient_id'], inplace=True)
    
    DISEASE_KEYWORDS_ORDERED = [
        # 상세 질병을 일반 질병보다 먼저 배치하여 우선순위 부여
        ("甲型肝炎", ["甲型肝炎", "甲肝"]), 
        ("乙型肝炎", ["乙型肝炎", "乙肝"]), 
        ("丙型肝炎", ["丙型肝炎", "丙肝"]),
        ("丁型肝炎", ["丁型肝炎", "丁肝"]), 
        ("戊型肝炎", ["戊型肝炎", "戊肝"]), 
        ("病毒性肝炎", ["病毒性肝炎", "肝炎"]), # '...肝炎'을 마지막에 배치

        ("人禽流感", ["人禽流感"]),
        ("人感染H7N9禽流感", ["H7N9"]), 
        ("流行性感冒", ["流行性感冒", "流感"]), # '流感' 키워드로 '流感样病例' 포함

        ("新型冠状病毒肺炎", ["新型冠状病毒肺炎", "冠状病毒", "新冠"]), 
        ("传染性非典", ["传染性非典", "非典", "SARS"]),
        
        ("细菌性和阿米巴性痢疾", ["细菌性和阿米巴性痢疾", "痢疾"]), 
        ("其他感染性腹泻病", ["其他感染性腹泻病", "腹泻病"]),

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
        ("急性出血性结膜炎", ["急性出血性结膜炎", "结膜炎"]), 
        ("脊灰", ["脊灰"]), 
        ("狂犬病", ["狂犬病"]),
        ("淋病", ["淋病"]), 
        ("流行性腮腺炎", ["流行性腮腺炎", "腮腺炎"]), 
        ("流脑", ["流脑"]),
        ("麻风病", ["麻风病"]), 
        ("麻疹", ["麻疹"]), 
        ("梅毒", ["梅毒"]), 
        ("疟疾", ["疟疾"]),
        ("肺结核", ["肺结核", "结核"]), 
        ("伤寒和副伤寒", ["伤寒和副伤寒", "伤寒"]), 
        ("手足口病", ["手足口病", "手足口"]),
        ("鼠疫", ["鼠疫"]), 
        ("丝虫病", ["丝虫病"]), 
        ("血吸虫病", ["血吸虫病"]),
        ("新生儿破伤风", ["新生儿破伤风", "破伤风"]), 
        ("猩红热", ["猩红热"]), 
        ("乙脑", ["乙脑"]),
    ]
    ALL_DISEASES = [item[0] for item in DISEASE_KEYWORDS_ORDERED] + ['Other']


    def map_diagnosis_by_keyword_ordered(diagnosis):
        diag_str = str(diagnosis)
        for disease_name, keywords in DISEASE_KEYWORDS_ORDERED:
            for keyword in keywords:
                if keyword in diag_str:
                    return disease_name
        return 'Other'

    df['epi_category'] = df['primary_diagnosis'].apply(map_diagnosis_by_keyword_ordered)

    # --- 주간 통계 집계 (고유 환자 수) ---
    df_epi = df.set_index('diagnosis_time').groupby(
        [pd.Grouper(freq='W-MON'), 'epi_category']
    )['outpatient_id'].nunique().unstack(fill_value=0)
    
    df_epi.columns.name = None
    df_epi = df_epi.reset_index()

    # 모든 질병 컬럼이 존재하도록 보장
    for disease in ALL_DISEASES:
        if disease not in df_epi.columns:
            df_epi[disease] = 0
    
    # 컬럼 순서 정리
    first_col = 'diagnosis_time'
    sorted_cols = [first_col] + sorted([col for col in df_epi.columns if col != first_col])
    df_epi = df_epi[sorted_cols]

    # --- 주간 통계 집계 (고유 환자 수 기준 Top 20 질병) ---
    # 전체 기간 동안 '고유 환자 수' 기준 상위 20개 질병 선정
    try:
        total_unique_by_cat = (
            df.groupby('epi_category')['outpatient_id'].nunique().sort_values(ascending=False)
        )
        top20_categories = total_unique_by_cat.head(20).index.tolist()
    except Exception:
        top20_categories = []

    # 주간 × 질병 고유 환자 수 피벗은 이미 df_epi가 충족하므로 거기서 Top20만 서브셋
    if top20_categories:
        for cat in top20_categories:
            if cat not in df_epi.columns:
                df_epi[cat] = 0
        cols = ['diagnosis_time'] + top20_categories
        df_weekly_top20_unique = df_epi[cols].copy()
    else:
        df_weekly_top20_unique = df_epi[['diagnosis_time']].copy()

    # 동일 Top20 질병에 대해 '주간 방문 건수(사이즈)'로 집계한 표도 생성
    df_weekly_visit_counts = df.set_index('diagnosis_time').groupby(
        [pd.Grouper(freq='W-MON'), 'epi_category']
    )['outpatient_id'].size().unstack(fill_value=0)
    df_weekly_visit_counts.columns.name = None
    df_weekly_visit_counts = df_weekly_visit_counts.reset_index()
    if top20_categories:
        for cat in top20_categories:
            if cat not in df_weekly_visit_counts.columns:
                df_weekly_visit_counts[cat] = 0
        df_weekly_top20_visits = df_weekly_visit_counts[['diagnosis_time'] + top20_categories].copy()
    else:
        df_weekly_top20_visits = df_weekly_visit_counts[['diagnosis_time']].copy()

    return df_epi, total_unique_patients, top_10_departments, df_weekly_top20_unique, df_weekly_top20_visits

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(base_dir, 'data', 'HIS门诊.xlsx')
    output_dir = os.path.join(base_dir, 'processed_data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    his_epi_stats, unique_patients, top_depts, his_top20_unique_stats, his_top20_visit_stats = process_his_outpatient_data(data_file)

    if his_epi_stats is not None:
        output_file = os.path.join(output_dir, 'his_outpatient_weekly_epi_counts.csv')
        his_epi_stats.to_csv(output_file, index=False, encoding='utf-8-sig')
        # Top20(고유 환자 기준 선정) 주간 통계 - 고유 환자 수
        output_file_top20_unique = os.path.join(output_dir, 'his_outpatient_weekly_epi_top20_unique_counts.csv')
        his_top20_unique_stats.to_csv(output_file_top20_unique, index=False, encoding='utf-8-sig')
        # Top20(동일 목록) 주간 통계 - 방문 건수(사이즈)
        output_file_top20_visits = os.path.join(output_dir, 'his_outpatient_weekly_epi_top20_visit_counts.csv')
        his_top20_visit_stats.to_csv(output_file_top20_visits, index=False, encoding='utf-8-sig')
        
        print("### HIS 외래 데이터 분석 결과 ###")
        print(f"\n1. 총 고유 환자 수: {unique_patients}명")
        
        print("\n2. 상위 10개 진료과 (건수):")
        print(top_depts.to_string())
        
        print(f"\n3. 주간 전염병 통계(고유 환자 수)가 다음 파일에 저장되었습니다: {output_file}")
        print(f"4. 주간 전염병 통계(Top20·고유 환자 수)가 다음 파일에 저장되었습니다: {output_file_top20_unique}")
        print(f"5. 주간 전염병 통계(Top20·방문 건수)가 다음 파일에 저장되었습니다: {output_file_top20_visits}")
        print("\n[주간 통계 결과 샘플]")
        print(his_epi_stats.head())

