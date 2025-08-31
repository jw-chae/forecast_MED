from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from .adapters import load_his_outpatient_series
from .fusion import precision_weighted_fusion
from .scenario_engine import extract_growth_episodes, generate_paths_conditional
from .evt import fit_pot, replace_tail_with_evt


def compute_forecast(series: np.ndarray, horizon: int = 8, seed: int = 123, news: float = 0.35, quality: float = 0.72):
    y = series.astype(float)
    yhat_mean = float(np.mean(y[-8:])) if len(y) >= 8 else float(np.mean(y))
    yhat_var = float(np.var(y[-16:])) if len(y) >= 16 else max(1.0, float(np.var(y)))

    fusion_res = precision_weighted_fusion(
        yhat_mean=yhat_mean,
        yhat_var=yhat_var,
        y_obs=float(y[-1]),
        data_quality=quality,
        manual_bias_mean=0.20,
        manual_bias_sd=0.10,
        news_signal=news,
    )

    episodes = extract_growth_episodes(y)
    paths = generate_paths_conditional(
        series=y,
        horizon=horizon,
        n_paths=5000,
        episodes=episodes,
        news_signal=news,
        quality=quality,
        random_state=seed,
    )

    u = float(np.quantile(y, 0.9))
    gpd = fit_pot(y, threshold=u)
    paths_evt = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)

    qs = [0.5, 0.8, 0.95]
    quantiles = {f"q{int(q*100)}": np.quantile(paths_evt, q, axis=0).tolist() for q in qs}
    mean_path = paths_evt.mean(axis=0).tolist()

    return {
        "fusion": fusion_res.as_dict(),
        "mean_path": mean_path,
        "quantiles": quantiles,
    }


def build_html(dates: List[str], values: List[float], future_dates: List[str], forecast: dict, disease: str) -> str:
    # Plotly CDN 사용(간단). 필요시 로컬 파일 배포로 교체 가능
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"

    q50 = forecast["quantiles"]["q50"]
    q80 = forecast["quantiles"]["q80"]
    q95 = forecast["quantiles"]["q95"]
    mean_path = forecast["mean_path"]
    fusion = forecast["fusion"]

    data_json = json.dumps({
        "dates": dates,
        "values": values,
        "future_dates": future_dates,
        "q50": q50,
        "q80": q80,
        "q95": q95,
        "mean_path": mean_path,
        "fusion": fusion,
        "disease": disease,
    }, ensure_ascii=False)

    html = f"""
<!DOCTYPE html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\"/>
  <title>HIS 외래 예측 리포트</title>
  <script src=\"{plotly_cdn}\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    #chart {{ width: 100%; height: 600px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    .pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; margin-right: 8px; background:#f3f4f6; }}
  </style>
</head>
<body>
  <h2>HIS 외래 주간 통계 예측 · <span id=\"disease\"></span></h2>
  <div class=\"card\">
    <div class=\"pill\" id=\"nowcast\"></div>
    <div class=\"pill\" id=\"ci\"></div>
  </div>
  <div id=\"chart\"></div>
  <script>
    const d = {data_json};
    document.getElementById('disease').textContent = d.disease;
    document.getElementById('nowcast').textContent = 'nowcast: ' + Number(d.fusion.mean).toFixed(2);
    document.getElementById('ci').textContent = '95% CI: [' + Number(d.fusion.ci95[0]).toFixed(2) + ', ' + Number(d.fusion.ci95[1]).toFixed(2) + ']';

    const traceHist = {{
      x: d.dates,
      y: d.values,
      name: '원본(주간 고유 환자수)',
      mode: 'lines+markers',
      line: {{color: '#1f77b4'}},
      marker: {{size: 4}}
    }};

    const traceMean = {{
      x: d.future_dates,
      y: d.mean_path,
      name: '예측(평균)',
      mode: 'lines',
      line: {{color: '#d62728', width: 2}}
    }};

    const band95 = {{
      x: d.future_dates.concat([...d.future_dates].reverse()),
      y: d.q95.concat([...d.q50].reverse()),
      fill: 'toself',
      fillcolor: 'rgba(214,39,40,0.10)',
      line: {{color: 'rgba(0,0,0,0)'}},
      name: '50–95% band'
    }};
    const band80 = {{
      x: d.future_dates.concat([...d.future_dates].reverse()),
      y: d.q80.concat([...d.q50].reverse()),
      fill: 'toself',
      fillcolor: 'rgba(214,39,40,0.20)',
      line: {{color: 'rgba(0,0,0,0)'}},
      name: '50–80% band'
    }};
    const traceMedian = {{
      x: d.future_dates,
      y: d.q50,
      name: '예측(중앙값)',
      mode: 'lines',
      line: {{color: '#d62728', dash: 'dash'}}
    }};

    const layout = {{
      hovermode: 'x unified',
      xaxis: {{title: '주(월요일 기준)'}},
      yaxis: {{title: '주간 고유 환자수'}},
      legend: {{orientation: 'h', y: -0.2}},
      margin: {{t: 20, r: 20, b: 80, l: 60}}
    }};
    Plotly.newPlot('chart', [traceHist, band95, band80, traceMedian, traceMean], layout, {{displaylogo:false}});
  </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="流行性感冒")
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    dt_index, series = load_his_outpatient_series(str(csv_path), args.disease)

    # 데이터가 비어 있거나 모두 0인 경우 처리
    if series.size == 0 or np.all(series == 0):
        print(f"Error: No data or only zero values found for disease '{args.disease}'. Cannot generate forecast.")
        return

    forecast = compute_forecast(series, horizon=args.horizon)

    # 날짜 문자열 변환
    dates = [d.strftime("%Y-%m-%d") for d in dt_index]
    last = dt_index.iloc[-1]
    future_dates = [(last + np.timedelta64(7 * (i + 1), 'D')).strftime("%Y-%m-%d") for i in range(args.horizon)]

    html = build_html(dates, series.astype(float).tolist(), future_dates, forecast, args.disease)

    out_dir = base / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"his_outpatient_forecast_{args.disease}.html"
    out_file.write_text(html, encoding="utf-8")
    print(str(out_file))


if __name__ == "__main__":
    main()


