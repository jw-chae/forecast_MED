from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .adapters import load_his_outpatient_series
from .scenario_engine import extract_growth_episodes, generate_paths_conditional
from .evt import fit_pot, replace_tail_with_evt
from .evidence_pack import build_evidence_pack_from_gov_monthly_csv, map_evidence_to_param_hints


def iso(s: str) -> str:
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def build_html(dates_hist, values_hist, dates_target, values_target, future_dates, forecast, disease, train_until, end_date) -> str:
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    data_json = json.dumps({
        "dates_hist": dates_hist,
        "values_hist": values_hist,
        "dates_target": dates_target,
        "values_target": values_target,
        "future_dates": future_dates,
        "q50": forecast["quantiles"]["q50"],
        "q80": forecast["quantiles"]["q80"],
        "q95": forecast["quantiles"]["q95"],
        "mean_path": forecast["mean_path"],
        "fusion": forecast["fusion"],
        "disease": disease,
        "train_until": train_until,
        "end_date": end_date,
    }, ensure_ascii=False)

    html_tpl = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>Holdout Forecast · __TITLE__</title>
  <script src="__PLOTLY__"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
    #chart { width: 100%; height: 640px; }
    .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; background:#f3f4f6; margin-right: 8px; }
  </style>
</head>
<body>
  <h2>HIS 외래 Holdout 예측 · <span id="disease"></span></h2>
  <div>
    <span class="pill" id="split"></span>
    <span class="pill" id="nowcast"></span>
    <span class="pill" id="ci"></span>
    <span class="pill" id="metrics"></span>
  </div>
  <div id="chart"></div>
  <script>
    const d = __DATA_JSON__;
    document.getElementById('disease').textContent = d.disease;
    document.getElementById('split').textContent = 'train≤ ' + d.train_until + ' → forecast to ' + d.end_date;
    document.getElementById('nowcast').textContent = 'nowcast: ' + Number(d.fusion.mean).toFixed(2);
    document.getElementById('ci').textContent = '95% CI: [' + Number(d.fusion.ci95[0]).toFixed(2) + ', ' + Number(d.fusion.ci95[1]).toFixed(2) + ']';

    function mae(a,b) { let s=0; let n=0; for(let i=0;i<Math.min(a.length,b.length);i++){ if(a[i]!=null&&b[i]!=null){ s+=Math.abs(a[i]-b[i]); n++; } } return n? s/n : null; }
    function coverage(lo, hi, y) { let n=0; let c=0; for(let i=0;i<Math.min(lo.length, y.length); i++){ if(y[i]!=null){ n++; if(y[i]>=lo[i] && y[i]<=hi[i]) c++; } } return n? c/n : null; }
    const mae50 = mae(d.q50, d.values_target);
    const cov95 = coverage(d.q50.map((m,i)=>Math.min(m,d.q95[i])), d.q95, d.values_target);
    document.getElementById('metrics').textContent = 'MAE(median)=' + (mae50? mae50.toFixed(2): '-') + ', 95% coverage=' + (cov95? cov95.toFixed(2): '-');

    const traceHist = { x: d.dates_hist, y: d.values_hist, name: '학습기간', mode: 'lines', line: {color:'#555'} };
    const traceTarget = { x: d.dates_target, y: d.values_target, name: '실제(2023–2024)', mode: 'lines+markers', line: {color:'#1f77b4'}, marker:{size:4} };
    const traceMean = { x: d.future_dates, y: d.mean_path, name: '예측(평균)', mode: 'lines', line: {color:'#d62728'} };
    const band95 = { x: d.future_dates.concat([...d.future_dates].reverse()), y: d.q95.concat([...d.q50].reverse()), fill: 'toself', fillcolor:'rgba(214,39,40,0.10)', line:{color:'rgba(0,0,0,0)'}, name:'50–95% band' };
    const band80 = { x: d.future_dates.concat([...d.future_dates].reverse()), y: d.q80.concat([...d.q50].reverse()), fill: 'toself', fillcolor:'rgba(214,39,40,0.20)', line:{color:'rgba(0,0,0,0)'}, name:'50–80% band' };
    const traceMedian = { x: d.future_dates, y: d.q50, name: '예측(중앙값)', mode: 'lines', line: {color:'#d62728', dash:'dash'} };
    const layout = { hovermode:'x unified', xaxis:{title:'주(월요일 기준)'}, yaxis:{title:'주간 고유 환자수'}, legend:{orientation:'h', y:-0.2}, margin:{t:20,r:20,b:80,l:60} };
    Plotly.newPlot('chart', [traceHist, traceTarget, band95, band80, traceMedian, traceMean], layout, {displaylogo:false});
  </script>
</body>
</html>
"""
    html = (
        html_tpl.replace("__PLOTLY__", plotly_cdn)
        .replace("__TITLE__", disease)
        .replace("__DATA_JSON__", data_json)
    )
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="手足口病")
    parser.add_argument("--train_until", default="2022-12-31")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--news", type=float, default=0.0)
    parser.add_argument("--quality", type=float, default=0.72)
    parser.add_argument("--gov_monthly_csv", default="", help="optional path to gov monthly stats CSV to derive news signal")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    dt_index, series = load_his_outpatient_series(str(csv_path), args.disease)
    dates = [d.strftime("%Y-%m-%d") for d in dt_index]

    # 학습/타깃 분할
    split_idx = max(i for i, d in enumerate(dates) if d <= iso(args.train_until))
    end_idx = max(i for i, d in enumerate(dates) if d <= iso(args.end))
    hist = series[: split_idx + 1]
    target = series[split_idx + 1 : end_idx + 1]
    horizon = len(target)

    # 예측 경로 생성
    episodes = extract_growth_episodes(hist)

    # 외부 신호(news):
    # 1) gov_monthly_csv 지정 시, 해당 CSV를 기반으로 as-of=train_until에서 스칼라 news_signal 도출
    #    → 홀드아웃 전 구간에 동일 적용(보수적). 없으면 (2) 계절성 의사-뉴스로 대체.
    from datetime import datetime as _dt
    def month_to_weight(ymd: str) -> float:
        m = _dt.strptime(ymd, "%Y-%m-%d").month
        # 독감(북반구) 시즌 priors: Nov–Feb HIGH, Oct MID, Mar–May LOW, Jun–Sep LOW–MID
        if m in (11, 12, 1, 2):
            return 0.7
        if m == 10:
            return 0.4
        if m in (3, 4, 5):
            return 0.08
        # 6–9월
        return 0.15
    future_dates = dates[split_idx + 1 : end_idx + 1]
    news_vec = None
    if args.gov_monthly_csv:
        try:
            pack = build_evidence_pack_from_gov_monthly_csv(args.gov_monthly_csv, asof=iso(args.train_until))
            hints = map_evidence_to_param_hints(pack)
            base_news = float(hints.get("news_signal", 0.1))
            news_vec = np.full(horizon, base_news, dtype=float)
        except Exception:
            news_vec = None
    if news_vec is None:
        news_vec = np.array([month_to_weight(d) for d in future_dates], dtype=float)

    # 시즌 기준선: 홀드아웃 첫 주들의 역사적 주차 중앙값을 초기값 하한으로 사용
    # 간단 근사: 학습 마지막 w주 중앙값을 사용
    season_start_override = float(np.median(hist[-8:]))

    paths = generate_paths_conditional(
        series=hist.astype(float),
        horizon=horizon,
        n_paths=5000,
        episodes=episodes,
        news_signal=news_vec,
        quality=args.quality,
        random_state=123,
        start_value_override=season_start_override,
        nb_dispersion_k=5.0,
    )
    # EVT 보정
    u = float(np.quantile(hist, 0.9))
    gpd = fit_pot(hist, threshold=u)
    paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)

    qs = [0.5, 0.8, 0.95]
    quantiles = {f"q{int(q*100)}": np.quantile(paths, q, axis=0).tolist() for q in qs}
    mean_path = paths.mean(axis=0).tolist()
    fusion = {"mean": float(hist[-1]), "variance": 1.0, "ci95": [float(hist[-1]), float(hist[-1])]}
    forecast = {"quantiles": quantiles, "mean_path": mean_path, "fusion": fusion}

    html = build_html(
        dates_hist=dates[: split_idx + 1],
        values_hist=hist.astype(float).tolist(),
        dates_target=future_dates,
        values_target=target.astype(float).tolist(),
        future_dates=future_dates,
        forecast=forecast,
        disease=args.disease,
        train_until=iso(args.train_until),
        end_date=iso(args.end),
    )

    out_dir = base / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"holdout_forecast_{args.disease}_{iso(args.train_until)}_{iso(args.end)}.html"
    out_file.write_text(html, encoding="utf-8")
    print(str(out_file))


if __name__ == "__main__":
    main()


