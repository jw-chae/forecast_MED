# -*- coding: utf-8 -*-
"""
HIS(외래) 및 LIS(진단검사) 데이터를 기반으로
주간 전염병 발생 통계를 인터랙티브 차트로 시각화합니다.

이 스크립트를 실행하기 전에 다음 라이브러리를 설치해야 합니다:
pip install pandas plotly
"""
import pandas as pd
import plotly.graph_objects as go
import os

def generate_interactive_chart(his_file, lis_file, output_html_file):
    """
    두 개의 데이터 소스를 통합하여 인터랙티브 질병 발생 차트를 생성하고 HTML 파일로 저장합니다.

    Args:
        his_file (str): HIS 외래 주간 통계 CSV 파일 경로.
        lis_file (str): LIS 진단검사 주간 통계 CSV 파일 경로.
        output_html_file (str): 생성될 HTML 파일 경로.
    """
    try:
        df_his = pd.read_csv(his_file)
        df_lis = pd.read_csv(lis_file)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e.filename}")
        return

    # 컬럼 이름 통일 및 데이터 소스 구분 컬럼 추가
    df_his.rename(columns={'diagnosis_time': 'date'}, inplace=True)
    df_lis.rename(columns={'INSPECTION_DATE': 'date'}, inplace=True)
    df_his['source'] = 'HIS(외래)'
    df_lis['source'] = 'LIS(진단검사)'

    # 날짜 형식 통일
    df_his['date'] = pd.to_datetime(df_his['date'])
    df_lis['date'] = pd.to_datetime(df_lis['date'])
    
    # 두 데이터프레임의 컬럼을 맞춤 (합집합으로 정렬, 결측은 0)
    disease_cols_his = [c for c in df_his.columns if c not in ('date', 'source')]
    disease_cols_lis = [c for c in df_lis.columns if c not in ('date', 'source')]
    all_diseases = sorted(list(set(disease_cols_his) | set(disease_cols_lis)))
    # 누락된 질병 컬럼 추가 후 0으로 채움
    for c in all_diseases:
        if c not in df_his.columns:
            df_his[c] = 0
        if c not in df_lis.columns:
            df_lis[c] = 0
    df_his = df_his[['date', 'source'] + all_diseases]
    df_lis = df_lis[['date', 'source'] + all_diseases]

    # 데이터 통합 및 Long-form 으로 변환
    df_combined = pd.concat([df_his, df_lis], ignore_index=True)
    df_long = df_combined.melt(
        id_vars=['date', 'source'], 
        var_name='disease', 
        value_name='count'
    )
    
    # 'Other'는 제외하고, 발생 건수가 0 이상인 데이터만 사용
    df_long = df_long[(df_long['disease'] != 'Other') & (df_long['count'] > 0)]

    # --- 인터랙티브 차트 생성 ---
    fig = go.Figure()

    # 데이터 소스별, 질병별로 라인 추가
    for source in ['HIS(외래)', 'LIS(진단검사)']:
        for disease in sorted(df_long['disease'].unique()):
            df_plot = df_long[(df_long['source'] == source) & (df_long['disease'] == disease)]
            if not df_plot.empty:
                fig.add_trace(go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['count'],
                    name=f"{disease} ({source})",
                    mode='lines+markers',
                    visible=False # 기본적으로는 보이지 않게 설정
                ))

    # 기본적으로 '유행성감기'는 보이도록 설정
    for i, trace in enumerate(fig.data):
        if '流行性感冒' in trace.name:
            fig.data[i].visible = True

    # --- 드롭다운 메뉴 및 버튼 추가 ---
    buttons = []
    # 각 질병별로 버튼 생성
    for disease in sorted(df_long['disease'].unique()):
        visibility = [(disease in trace.name) for trace in fig.data]
        buttons.append(dict(
            label=disease,
            method='update',
            args=[{'visible': visibility}, {'title': f'{disease} 주간 발생 현황'}]
        ))
    
    # 전체 보기 버튼 추가
    buttons.insert(0, dict(
        label='전체 보기',
        method='update',
        args=[{'visible': [True] * len(fig.data)}, {'title': '전체 질병 주간 발생 현황'}]
    ))
    
    # 초기화(유행성감기) 버튼 추가
    buttons.insert(1, dict(
        label='초기화 (독감)',
        method='update',
        args=[{'visible': ['流行性感冒' in trace.name for trace in fig.data]}, {'title': '流行性感冒 주간 발생 현황'}]
    ))


    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )],
        title_text='流行性感冒 주간 발생 현황',
        xaxis_title='날짜',
        yaxis_title='주간 고유 환자 수',
        autosize=True,
        height=700,
        hovermode='x unified'
    )

    # HTML 파일로 저장
    fig.write_html(output_html_file)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    his_data_file = os.path.join(base_dir, 'processed_data', 'his_outpatient_weekly_epi_counts.csv')
    # LIS: 진단 기반 Top20 + 비법정 상위4 포함(고유 환자 수)
    lis_data_file = os.path.join(base_dir, 'processed_data', 'lis_weekly_epi_top20_plus4_unique_counts.csv')
    
    output_file = os.path.join(base_dir, 'interactive_disease_chart.html')
    
    generate_interactive_chart(his_data_file, lis_data_file, output_file)
    
    print("### 인터랙티브 차트 생성 완료 ###")
    print(f"결과가 다음 파일에 저장되었습니다:\n{output_file}")
    print("\n브라우저에서 위 파일을 열어 확인해주세요.")

