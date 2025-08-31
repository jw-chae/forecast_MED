from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

from adapters import load_his_outpatient_series
from fusion import precision_weighted_fusion
from scenario_engine import extract_growth_episodes, generate_paths_conditional
from evt import fit_pot, replace_tail_with_evt
from risk_banding import band_from_paths

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def main() -> None:
    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    output_dir = base / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # 전체 인플루엔자 데이터 로드
    dates, series = load_his_outpatient_series(str(csv_path), "流行性感冒")
    
    # 2024년 1월-6월 데이터만 필터링
    df = pd.DataFrame({'date': dates, 'count': series})
    df_2024_h1 = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-06-30')]
    
    y = df_2024_h1['count'].values.astype(float)
    dates_2024_h1 = df_2024_h1['date'].values
    
    # 예측 분석
    yhat_mean = float(np.mean(y[-8:])) if len(y) >= 8 else float(np.mean(y))
    yhat_var = float(np.var(y[-16:])) if len(y) >= 16 else max(1.0, float(np.var(y)))
    
    fusion_res = precision_weighted_fusion(
        yhat_mean=yhat_mean,
        yhat_var=yhat_var,
        y_obs=float(y[-1]),
        data_quality=0.72,
        manual_bias_mean=0.20,
        manual_bias_sd=0.10,
        news_signal=0.35,
    )
    
    episodes = extract_growth_episodes(y)
    paths = generate_paths_conditional(
        series=y,
        horizon=8,
        n_paths=2000,
        episodes=episodes,
        news_signal=0.35,
        quality=0.72,
        random_state=123,
    )
    
    if len(y) > 10:
        u = float(np.quantile(y, 0.9))
        gpd = fit_pot(y, threshold=u)
        paths_evt = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)
    else:
        paths_evt = paths
    
    risk = band_from_paths(
        paths_evt,
        current_level=float(y[-1]),
        er_wait_baseline_min=72.0,
        bed_occupancy_baseline=0.84,
    )
    
    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('2024년 1-6월 인플루엔자 분석 결과', fontsize=16, fontweight='bold')
    
    # 1. 시계열 데이터
    ax1.plot(dates_2024_h1, y, 'b-o', linewidth=2, markersize=4, label='실제 환자 수')
    ax1.axhline(y=np.mean(y), color='r', linestyle='--', alpha=0.7, label=f'평균: {np.mean(y):.1f}')
    ax1.axhline(y=np.max(y), color='orange', linestyle='--', alpha=0.7, label=f'최대: {np.max(y):.0f}')
    ax1.set_title('주별 인플루엔자 환자 수')
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('환자 수')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 예측 경로들
    future_weeks = 8
    last_date = dates_2024_h1[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq='W')
    
    # 경로들의 분위수 계산
    q05 = np.percentile(paths_evt, 5, axis=0)
    q25 = np.percentile(paths_evt, 25, axis=0)
    q50 = np.percentile(paths_evt, 50, axis=0)
    q75 = np.percentile(paths_evt, 75, axis=0)
    q95 = np.percentile(paths_evt, 95, axis=0)
    
    # 과거 데이터
    ax2.plot(dates_2024_h1, y, 'b-o', linewidth=2, markersize=4, label='과거 데이터')
    
    # 예측 구간
    ax2.fill_between(future_dates, q05, q95, alpha=0.2, color='red', label='90% 예측구간')
    ax2.fill_between(future_dates, q25, q75, alpha=0.3, color='orange', label='50% 예측구간')
    ax2.plot(future_dates, q50, 'r-', linewidth=2, label='중위수 예측')
    
    ax2.set_title('8주 예측 결과')
    ax2.set_xlabel('날짜')
    ax2.set_ylabel('환자 수')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. 성장 에피소드
    ax3.plot(range(len(y)), y, 'b-', linewidth=2, alpha=0.7, label='환자 수')
    
    # 에피소드 표시
    colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray']
    for i, episode in enumerate(episodes):
        color = colors[i % len(colors)]
        start_idx = episode.start_idx
        end_idx = episode.end_idx
        peak_idx = start_idx + np.argmax(episode.values)
        
        ax3.axvspan(start_idx, end_idx, alpha=0.2, color=color, label=f'에피소드 {i+1}')
        ax3.plot(peak_idx, y[peak_idx], 'o', color=color, markersize=8)
    
    ax3.set_title(f'성장 에피소드 감지 (총 {len(episodes)}개)')
    ax3.set_xlabel('주차')
    ax3.set_ylabel('환자 수')
    if len(episodes) <= 7:
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 위험도 및 통계
    stats_text = f"""
위험도 평가: {risk.band}
확률: {risk.probability:.3f}

통계 정보:
• 평균: {np.mean(y):.1f}명
• 표준편차: {np.std(y):.1f}명
• 최대: {np.max(y):.0f}명
• 최소: {np.min(y):.0f}명

융합 분석:
• 예측 평균: {fusion_res.mean:.2f}
• 예측 표준편차: {fusion_res.std:.2f}
• 95% 신뢰구간: [{fusion_res.ci95[0]:.2f}, {fusion_res.ci95[1]:.2f}]

KPI 요약:
• 응급실 대기시간: {risk.kpi_summary['er_wait_mean']:.0f}분
• 병상 점유율: {risk.kpi_summary['bed_occupancy_mean']:.1f}%
• 임계값 초과 확률: {risk.kpi_summary['p_exceed']:.3f}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('분석 결과 요약')
    
    plt.tight_layout()
    
    # 저장
    output_file = output_dir / "influenza_2024_h1_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"시각화 결과가 저장되었습니다: {output_file}")
    
    plt.show()
    
    # 결과 JSON 저장
    result = {
        "period": "2024년 1월-6월",
        "disease": "인플루엔자 (流行性感冒)",
        "data_points": len(df_2024_h1),
        "date_range": {
            "start": df_2024_h1['date'].min().strftime('%Y-%m-%d'),
            "end": df_2024_h1['date'].max().strftime('%Y-%m-%d')
        },
        "statistics": {
            "mean": float(np.mean(y)),
            "max": float(np.max(y)),
            "min": float(np.min(y)),
            "std": float(np.std(y))
        },
        "fusion": fusion_res.as_dict(),
        "episodes_count": len(episodes),
        "risk": {
            "band": risk.band,
            "probability": risk.probability,
            "kpi_summary": risk.kpi_summary,
        },
        "prediction_quantiles": {
            "q05": q05.tolist(),
            "q25": q25.tolist(),
            "q50": q50.tolist(),
            "q75": q75.tolist(),
            "q95": q95.tolist()
        }
    }
    
    result_file = base / "reports" / "influenza_2024_h1_analysis.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"분석 결과가 저장되었습니다: {result_file}")
    print("\n=== 2024년 1월-6월 인플루엔자 분석 완료 ===")


if __name__ == "__main__":
    main()