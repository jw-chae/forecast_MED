from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import trange

from Tools.adapters import load_his_outpatient_series
from Tools.llm_agent import propose_params_via_llm, apply_hard_guards, clamp_params
from Tools.run_sim_wrapper import run_sim, SimConfig
from Tools.scenario_engine import extract_growth_episodes, generate_paths_conditional
from Tools.evt import fit_pot, replace_tail_with_evt
from Tools.metrics import interval_coverage, crps_gaussian, smape, mae, peak_metrics
from Tools.evidence_pack import (
    load_evidence_pack,
    build_evidence_pack_with_web,
    build_evidence_pack_with_weather,
    map_evidence_to_param_hints,
    build_evidence_pack_from_gov_monthly_csv,
)
from Tools.state_builder import build_observation as build_observation_v2
from Tools.config import APP_CONFIG


@dataclass
class RollingConfig:
    disease: str
    start_date: str
    end_date: str
    n_steps: int = 44
    horizon_weeks: int = 12
    season_profile: Optional[str] = "flu"


def dt(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def pick_step_dates(dates: List[str], start_date: str, end_date: str, n_steps: int) -> List[str]:
    # filter dates within range
    in_range = [d for d in dates if d >= start_date and d <= end_date]
    if not in_range:
        return []
    # n_steps <= 0 means use all weekly cutoffs
    if n_steps <= 0 or len(in_range) <= n_steps:
        return in_range
    # evenly spaced selection
    idxs = np.linspace(0, len(in_range) - 1, num=n_steps, dtype=int)
    return [in_range[i] for i in idxs]


def build_html(disease: str, steps: List[Dict[str, Any]], agg_pred_llm: Dict[str, List[Any]] | None = None, agg_pred_base: Dict[str, List[Any]] | None = None) -> str:
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    payload = json.dumps({"disease": disease, "steps": steps, "agg_llm": (agg_pred_llm or {}), "agg_base": (agg_pred_base or {})}, ensure_ascii=False)
    html_tpl = """
<!DOCTYPE html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\"/>
  <title>Weekly Rolling Forecast · __TITLE__</title>
  <script src=\"__PLOTLY__\"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
    #chart { width: 100%; height: 640px; }
    .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; background:#f3f4f6; margin-right: 8px; }
    select { padding:6px; }
  </style>
</head>
<body>
  <h2>주간 롤링 예측 · __TITLE__</h2>
  <div>
    <span class=\"pill\" id=\"info\"></span>
    <label>스텝 선택: <select id=\"stepSel\"></select></label>
    <span class=\"pill\" id=\"metrics\"></span>
  </div>
  <div id=\"chart\"></div>
  <script>
    const D = __PAYLOAD__;
    const steps = D.steps;
    const agg_llm = D.agg_llm || {};
    const agg_base = D.agg_base || {};
    const sel = document.getElementById('stepSel');
    steps.forEach((s,i)=>{ const o=document.createElement('option'); o.value=i; o.text=`${i+1} · train≤ ${s.train_until}`; sel.appendChild(o); });
    document.getElementById('info').textContent = `총 ${steps.length} 스텝, horizon=${(steps[0]?.horizon||0)}주`;
    function render(i){
      const s = steps[i];
      const traces = [];
      traces.push({ x: s.dates_hist, y: s.values_hist, name:'학습기간', mode:'lines', line:{color:'#999'} });
      traces.push({ x: s.dates_target, y: s.values_target, name:'실측(홀드아웃)', mode:'lines+markers', line:{color:'#1f77b4'}, marker:{size:4} });
      traces.push({ x: s.future_dates, y: s.quantiles.q50, name:'q50', mode:'lines', line:{dash:'dash', color:'#d62728'} });
      traces.push({ x: s.future_dates, y: s.quantiles.q95, name:'q95', mode:'lines', line:{color:'rgba(214,39,40,0.7)'} });
      traces.push({ x: s.future_dates, y: s.quantiles.q05, name:'q05', mode:'lines', line:{color:'rgba(214,39,40,0.7)', dash:'dot'} });
      if (agg_llm.pred_dates && agg_llm.q50 && agg_llm.q50.length){
        traces.push({ x: agg_llm.pred_dates, y: agg_llm.q50, name:'LLM pred q50 (weekly)', mode:'lines+markers', line:{color:'#d62728'}, marker:{size:4} });
        if (agg_llm.q05 && agg_llm.q95){
          traces.push({ x: agg_llm.pred_dates, y: agg_llm.q95, name:'LLM pred q95', mode:'lines', line:{color:'rgba(214,39,40,0.4)'} });
          traces.push({ x: agg_llm.pred_dates, y: agg_llm.q05, name:'LLM pred q05', mode:'lines', line:{color:'rgba(214,39,40,0.4)', dash:'dot'} });
        }
      }
      if (agg_base.pred_dates && agg_base.q50 && agg_base.q50.length){
        traces.push({ x: agg_base.pred_dates, y: agg_base.q50, name:'BASE pred q50', mode:'lines+markers', line:{color:'#9467bd'}, marker:{size:4} });
        if (agg_base.q05 && agg_base.q95){
          traces.push({ x: agg_base.pred_dates, y: agg_base.q95, name:'BASE pred q95', mode:'lines', line:{color:'rgba(148,103,189,0.5)'} });
          traces.push({ x: agg_base.pred_dates, y: agg_base.q05, name:'BASE pred q05', mode:'lines', line:{color:'rgba(148,103,189,0.5)', dash:'dot'} });
        }
      }
      const layout = { hovermode:'x unified', xaxis:{title:'week'}, yaxis:{title:'count'}, legend:{orientation:'h'}, margin:{t:20,r:20,b:80,l:60} };
      Plotly.newPlot('chart', traces, layout, {displaylogo:false});
      const m = s.metrics || {};
      const cov = (typeof m.coverage95==='number')? m.coverage95.toFixed(2): '-';
      const _crps = (typeof m.crps==='number')? m.crps.toFixed(2): '-';
      const rec = (typeof m.recall_pm2w==='number')? m.recall_pm2w.toFixed(2): '-';
      document.getElementById('metrics').textContent = `cov95=${cov}, CRPS=${_crps}, recall±2w=${rec}`;
    }
    sel.addEventListener('change', e=> render(parseInt(e.target.value)));
    if (steps.length) { sel.value = 0; render(0); }
  </script>
</body>
</html>
"""
    return html_tpl.replace("__PLOTLY__", plotly_cdn).replace("__TITLE__", disease).replace("__PAYLOAD__", payload)


def parse_args():
    parser = argparse.ArgumentParser()
    sim_defaults = APP_CONFIG.simulation
    parser.add_argument("--disease", type=str, default=sim_defaults.disease)
    parser.add_argument("--start", type=str, default=sim_defaults.start)
    parser.add_argument("--end", type=str, default=sim_defaults.end)
    parser.add_argument("--n_steps", type=int, default=sim_defaults.n_steps)
    parser.add_argument("--horizon", type=int, default=sim_defaults.horizon)
    parser.add_argument("--provider", type=str, default=sim_defaults.provider)
    parser.add_argument("--model", type=str, default=sim_defaults.model)
    parser.add_argument("--temperature", type=float, default=sim_defaults.temperature, help="LLM temperature")
    parser.add_argument("--train_len_weeks", type=int, default=sim_defaults.train_len_weeks, help="limit train history length")
    parser.add_argument("--no_llm", action="store_true", default=sim_defaults.no_llm)
    parser.add_argument("--save_json", action="store_true", default=sim_defaults.save_json)

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="手足口病")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--n_steps", type=int, default=44)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--csv_path", default="", help="원본 시계열 CSV 경로(기본: HIS 외래 주간) - LIS 대체 등에 사용")
    parser.add_argument("--evidence", default=None, help="증거팩(JSON 또는 디렉토리) 경로")
    parser.add_argument("--use_web", action="store_true", help="웹 뉴스/정부 공고 신호를 각 스텝 train_until 기준(as-of)으로 수집")
    parser.add_argument("--regions", default="中国 全国,浙江省", help="웹 검색 지역 콤마구분 문자열")
    parser.add_argument("--gov_only", action="store_true", help="정부 사이트 공고만 사용(뉴스 제외)")
    parser.add_argument("--site_whitelist", default="ndcpa.gov.cn,nhc.gov.cn,wjw.zj.gov.cn,wsjkw.zj.gov.cn,*.zj.gov.cn", help="사이트 화이트리스트 콤마구분")
    parser.add_argument("--region_keywords", default="全国,浙江省,全省", help="페이지 제목/본문에 포함돼야 하는 지역 키워드(하나 이상)")
    parser.add_argument("--gov_monthly_csv", default="", help="크롤링한 정부 월간 통계 CSV를 신호로 사용(웹 검색 대체)")
    parser.add_argument("--preset_aggr", action="store_true", help="봄 급등 구간 대응을 위한 보수 상향 프리셋 적용")
    parser.add_argument("--posthoc_cal", action="store_true", help="coverage 캘리브레이션 활성화")
    parser.add_argument("--target_cov", type=float, default=0.9, help="coverage 목표치 (0~1)")
    parser.add_argument("--params_json", default="", help="튜닝된 파라미터 JSON 경로(초기 파라미터로 사용)")
    parser.add_argument("--no_llm", action="store_true", help="LLM 제안 비활성화(초기/직전 파라미터만 사용)")
    parser.add_argument("--chain", type=int, default=0, help="체인 방식으로 k주 경로 생성(h=1을 k회 연결)")
    parser.add_argument("--chain_particles", type=int, default=500, help="체인 앙상블 파티클 수(기본 500)")
    parser.add_argument(
        "--provider",
        type=str,
        default="dashscope",
        choices=["dashscope", "openai"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model", type=str, default="qwen-max", help="LLM model name to use"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="LLM temperature"
    )
    parser.add_argument("--save_json", action="store_true", help="HTML과 함께 steps JSON 결과도 저장")
    parser.add_argument(
        "--train_len_weeks", type=int, default=None, help="limit train history length"
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    # progress log (txt) alongside HTML/JSON outputs
    out_dir = base / "reports" / "agent_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    import time, uuid
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    log_txt = out_dir / f"rolling_{args.disease}_{args.start}_{args.end}_{args.n_steps}x{args.horizon}_{run_id}.log"
    def _log(msg: str) -> None:
        try:
            from datetime import datetime as _dt
            with log_txt.open("a", encoding="utf-8") as f:
                f.write(f"[{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except Exception:
            pass
    # flexible loader to support alternative sources (e.g., LIS)
    def _load_series_generic(csv_path_str: str, disease: str):
        import pandas as _pd
        _df = _pd.read_csv(csv_path_str)
        if "diagnosis_time" in _df.columns:
            _dates = _pd.to_datetime(_df["diagnosis_time"]).dt.strftime("%Y-%m-%d").tolist()
        elif "INSPECTION_DATE" in _df.columns:
            _dates = _pd.to_datetime(_df["INSPECTION_DATE"]).dt.strftime("%Y-%m-%d").tolist()
        else:
            raise ValueError("Unsupported date column: expected 'diagnosis_time' or 'INSPECTION_DATE'")
        if disease not in _df.columns:
            raise KeyError(f"Disease column not found in CSV: {disease}")
        _series = _df[disease].astype(float).values
        return _dates, _series

    _default_csv = str(base / "processed_data" / "his_outpatient_weekly_epi_counts.csv")
    csv_path = args.csv_path.strip() or _default_csv
    dates, series = _load_series_generic(csv_path, args.disease)

    if args.train_len_weeks:
        series = series[-args.train_len_weeks :]
        dates = dates[-args.train_len_weeks :]

    step_dates = pick_step_dates(dates, args.start, args.end, args.n_steps)
    steps: List[Dict[str, Any]] = []

    # rolling state
    last_params: Dict[str, Any] = {
        "amplitude_quantile": 0.9,
        "amplitude_multiplier": 1.8,
        "ratio_cap_quantile": 0.98,
        "warmup_weeks": 1,
        "use_delta_quantile": True,
        "delta_quantile": 0.05,
        "quality": 0.72,
        "nb_dispersion_k": 8.0,
        "r_boost_cap": 2.0,
        "scale_cap": 1.6,
        "x_cap_multiplier": 2.0,
        "evt_u_quantile": 0.9,
    }
    last_metrics: Dict[str, Any] = {}
    recent_metrics_window: List[Dict[str, Any]] = []
    constraints = {
        "bounds": {
            "amplitude_quantile": [0.85, 0.98],
            "amplitude_multiplier": [1.2, 2.8],
            "ratio_cap_quantile": [0.95, 0.999],
            "warmup_weeks": [0, 2],
            "delta_quantile": [0.01, 0.1],
            "quality": [0.5, 0.95],
            "nb_dispersion_k": [2.0, 50.0],
            "r_boost_cap": [1.2, 3.0],
            "scale_cap": [1.2, 1.8],
            "x_cap_multiplier": [1.5, 4.0],
            "evt_u_quantile": [0.85, 0.95],
        }
    }

    regions = [r.strip() for r in (args.regions or "").split(",") if r.strip()]
    site_whitelist = [s.strip() for s in (args.site_whitelist or "").split(",") if s.strip()]
    region_keywords = [k.strip() for k in (args.region_keywords or "").split(",") if k.strip()]

    # optional: load tuned params as starting point
    if args.params_json:
        try:
            import json
            obj = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
            cand = obj.get("params", obj)
            if isinstance(cand, dict):
                last_params.update({k: v for k, v in cand.items() if k in last_params or k in (
                    "quality","nb_dispersion_k","delta_quantile","r_boost_cap","scale_cap","x_cap_multiplier","evt_u_quantile")})
        except Exception:
            pass

    # optional aggressive preset for spring surge
    if args.preset_aggr:
        last_params.update({
            "amplitude_multiplier": 2.4,
            "ratio_cap_quantile": 0.99,
            "warmup_weeks": 0,
            "quality": 0.69,
            "nb_dispersion_k": 18.0,
            "delta_quantile": 0.08,
            "r_boost_cap": 2.3,
        })

    for i, train_until in enumerate(step_dates):
        _log(f"step {i+1}/{len(step_dates)} · train_until={train_until}")
        
        # horizon 동적 조정: 예측 종료일이 전체 종료일을 넘지 않도록
        train_until_dt = dt(train_until)
        end_date_dt = min(dt(args.end), train_until_dt + timedelta(weeks=args.horizon))
        horizon = (end_date_dt - train_until_dt).days // 7
        
        if horizon <= 0:
            continue

        end_date = iso(end_date_dt)

        # evidence: load base then augment with web(as-of)
        ev = None
        if args.evidence:
            try:
                ev = load_evidence_pack(args.evidence)
            except Exception:
                ev = None
        if args.gov_monthly_csv:
            try:
                # news weekly vector for horizon weeks with longer decay
                ev = build_evidence_pack_from_gov_monthly_csv(
                    args.gov_monthly_csv,
                    base=(ev or {}),
                    asof=train_until,
                    future_weeks=args.horizon,
                    future_decay=[1.0,0.95,0.9,0.85,0.8,0.7,0.6,0.5][:max(1, args.horizon)],
                )
            except Exception:
                pass
        elif args.use_web:
            try:
                ev = build_evidence_pack_with_web(
                    args.disease,
                    regions,
                    base=(ev or {}),
                    asof=train_until,
                    gov_only=args.gov_only,
                    site_whitelist=site_whitelist,
                    region_keywords=region_keywords,
                )
            except Exception:
                # 웹 호출 실패 시에도 계속 진행
                pass
        
        # Add weather data to evidence pack
        ev = build_evidence_pack_with_weather(base=(ev or {}), asof=train_until, location="hangzhou")

        # previous LLM pred error from last appended step (if any)
        last_llm_pred_q50 = None
        last_llm_abs_err = None
        if steps:
            try:
                prev_step = steps[-1]
                if prev_step.get("quantiles") and prev_step.get("values_target"):
                    q50_prev = float(prev_step["quantiles"]["q50"][0])
                    y_prev = float(prev_step["values_target"][0])
                    last_llm_pred_q50 = q50_prev
                    last_llm_abs_err = abs(y_prev - q50_prev)
            except Exception:
                pass

        obs = build_observation_v2(
            disease=args.disease,
            train_until=train_until,
            end_date=iso(end_date_dt),
            last_params=last_params,
            last_metrics=last_metrics,
            constraints=constraints,
            evidence=ev,
            recent_metrics_window=recent_metrics_window,
            last_llm_pred_q50=last_llm_pred_q50,
            last_llm_abs_err=last_llm_abs_err,
        )

        llm_rationale = None
        llm_proposal = None
        if args.no_llm:
            _p = None
        else:
            try:
                llm_response = propose_params_via_llm(
                    obs,
                    provider=args.provider,
                    model=args.model,
                    temperature=args.temperature,
                )
                _p = llm_response["proposal"]
                llm_rationale = llm_response.get("rationale")
                llm_proposal = llm_response.get("proposal")

            except Exception as e:
                print(f"LLM call failed: {e}")
                _p = None
        p = _p or dict(last_params)
        p = apply_hard_guards(p)
        p = clamp_params(p, constraints)

        # adaptive backoff when LLM fails or when prior performance indicates overprediction/overcoverage
        try:
            cov = last_metrics.get("coverage95") if isinstance(last_metrics, dict) else None
            recent_growth_pct = obs.get("recent_growth_pct", 0.0)
            need_backoff = (_p is None) or (isinstance(cov, (int, float)) and float(cov) >= 0.98) or (isinstance(last_llm_abs_err, (int, float)) and last_llm_abs_err > 8.0 and recent_growth_pct <= 0.0)
            if need_backoff:
                p["amplitude_multiplier"] = float(min(p.get("amplitude_multiplier", 1.8), 1.4))
                p["r_boost_cap"] = float(min(p.get("r_boost_cap", 2.0), 1.6))
                p["scale_cap"] = float(min(p.get("scale_cap", 1.6), 1.4))
                p["x_cap_multiplier"] = float(min(p.get("x_cap_multiplier", 2.0), 1.8))
                # widen band slightly but keep within bounds
                q = float(p.get("quality", 0.72))
                p["quality"] = float(max(0.6, min(0.8, q)))
                _log("  adaptive_backoff applied")
                p = apply_hard_guards(p)
                p = clamp_params(p, constraints)
        except Exception:
            pass
        # If evidence exists and news_signal missing, inject from hints
        if ev is not None and (p.get("news_signal") is None):
            try:
                hints = map_evidence_to_param_hints(ev)
                # deterministic mapping fallback using signals if present
                if isinstance(hints.get("news_signal"), (int, float)):
                    ns = float(hints["news_signal"]) 
                else:
                    ext = (ev or {}).get("external_signals", {})
                    chg = float(ext.get("news_hits_change_4w", 0.0) or 0.0)
                    snr = float(ext.get("search_snr", 0.0) or 0.0)
                    ns = max(0.05, min(0.7, 0.05 + 0.25 * chg + 0.3 * (snr / 3.0)))
                p["news_signal"] = ns
                p = apply_hard_guards(p)
                p = clamp_params(p, constraints)
            except Exception:
                pass
        # pass evidence to params for weekly vector usage and enable post-hoc calibration if requested
        if args.posthoc_cal:
            p["enable_posthoc_calibration"] = True
            p["calibrate_coverage_to"] = float(max(0.5, min(0.98, args.target_cov)))
        if ev:
            p["evidence"] = ev
        # chain mode: build k-week path by iterating 1-week ahead predictions
        if args.chain and args.chain > 0:
            k = int(args.chain)
            # base data
            base = Path(__file__).resolve().parents[2]
            csv_path2 = args.csv_path.strip() or str(base / "processed_data" / "his_outpatient_weekly_epi_counts.csv")
            # reuse generic loader to support alternative CSV
            dates_all, series_all = _load_series_generic(csv_path2, args.disease)
            split_idx = max(i for i, d in enumerate(dates_all) if d <= train_until)
            hist = series_all[: split_idx + 1].astype(float)
            # available target length bound by args.end and k weeks after train_until
            from datetime import timedelta as _td
            end_lim = min(dt(args.end), dt(train_until) + _td(weeks=k))
            end_idx_lim = max(i for i, d in enumerate(dates_all) if d <= iso(end_lim))
            k_use = max(0, end_idx_lim - split_idx)
            if k_use <= 0:
                continue
            # news vector: evidence weekly if available, else scalar
            news_vec = None
            if isinstance(p.get("evidence", {}).get("external_signals", {}).get("news_signal_weekly"), list):
                nv = p["evidence"]["external_signals"]["news_signal_weekly"]
                if nv:
                    news_vec = [float(nv[i] if i < len(nv) else nv[-1]) for i in range(k_use)]
                else:
                    news_vec = [float(p.get("news_signal", 0.1))] * k_use
            else:
                news_vec = [float(p.get("news_signal", 0.1))] * k_use
            # ensemble chain: propagate particle set
            n_particles = max(100, int(args.chain_particles))
            particles = np.full(n_particles, float(hist[-1]))
            q05_list, q50_list, q95_list = [], [], []
            y = hist.copy()
            for step in range(k_use):
                episodes = extract_growth_episodes(y)
                season_start_override = float(np.median(y[-8:]))
                # draw 1w-ahead for each particle by resampling paths' terminal values
                paths = generate_paths_conditional(
                    series=y,
                    horizon=1,
                    n_paths=n_particles,
                    episodes=episodes,
                    news_signal=float(news_vec[step]),
                    quality=float(p.get("quality", 0.68)),
                    recent_baseline_window=int(p.get("recent_baseline_window", 8)),
                    amplitude_quantile=float(p.get("amplitude_quantile", 0.9)),
                    amplitude_multiplier=float(p.get("amplitude_multiplier", 2.2)),
                    ratio_cap_quantile=float(p.get("ratio_cap_quantile", 0.99)),
                    warmup_weeks=0,
                    use_delta_quantile=bool(p.get("use_delta_quantile", True)),
                    delta_quantile=float(p.get("delta_quantile", 0.05)),
                    nb_dispersion_k=(None if p.get("nb_dispersion_k") is None else float(p.get("nb_dispersion_k"))),
                    start_value_override=float(p.get("start_value_override", season_start_override)),
                    r_boost_cap=float(p.get("r_boost_cap", 2.0)),
                    scale_cap=float(p.get("scale_cap", 1.6)),
                    x_cap_multiplier=float(p.get("x_cap_multiplier", 2.0)),
                )
                # EVT polish
                u = float(np.quantile(y, float(p.get("evt_u_quantile", 0.9))))
                gpd = fit_pot(y, threshold=u, min_excess=int(p.get("min_excess", 5)))
                paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)
                # terminal distribution (shape: n_particles)
                # non-negative terminal values
                term = np.maximum(paths[:, -1], 0.0)
                q05_list.append(float(np.quantile(term, 0.05)))
                q50_list.append(float(np.quantile(term, 0.50)))
                q95_list.append(float(np.quantile(term, 0.95)))
                # resample particles for next step
                idx = np.random.choice(term.shape[0], size=n_particles, replace=True)
                particles = term[idx]
                y = np.append(y, float(np.median(particles)))
            # build synthetic res like run_sim
            # target and dates
            all_dates = dates
            split_idx2 = max(i for i, d in enumerate(all_dates) if d <= train_until)
            future_dates = all_dates[split_idx2 + 1 : split_idx2 + 1 + k_use]
            target = series.astype(float)[split_idx2 + 1 : split_idx2 + 1 + k_use]
            q05 = np.array(q05_list); q50 = np.array(q50_list); q95 = np.array(q95_list)
            # posthoc calibration if requested
            if args.posthoc_cal and len(target) == len(q50) and len(q50) > 0:
                desired = float(max(0.5, min(0.98, args.target_cov)))
                def cov_with_scale(s):
                    lo = q50 - s * (q50 - q05)
                    hi = q50 + s * (q95 - q50)
                    return float(np.mean((target >= lo) & (target <= hi)))
                cov0 = cov_with_scale(1.0)
                if cov0 < desired:
                    s_lo, s_hi = 1.0, 10.0
                    for _ in range(16):
                        s_mid = 0.5 * (s_lo + s_hi)
                        if cov_with_scale(s_mid) < desired: s_lo = s_mid
                        else: s_hi = s_mid
                    s_star = s_hi
                    q05 = q50 - s_star * (q50 - q05)
                    q95 = q50 + s_star * (q95 - q50)
            # metrics
            lower = np.minimum(q05, q95); upper = np.maximum(q05, q95)
            met = {
                "mae_median": float(mae(target, q50, use_median=True)),
                "smape": float(smape(target, q50)),
                "coverage95": float(interval_coverage(target, lower, upper)),
                "crps": float(crps_gaussian(target, q50, np.maximum(1e-6, (q95 - q05) / 3.92))),
            }
            met.update(peak_metrics(target, q50, alpha_top=float(p.get("peak_alpha", 0.1)), window_recall=2))
            res = {
                "horizon": int(k_use),
                "quantiles": {"q05": q05.tolist(), "q50": q50.tolist(), "q95": q95.tolist()},
                "future_dates": future_dates,
                "dates_hist": dates[: split_idx2 + 1],
                "values_hist": series.astype(float)[: split_idx2 + 1].tolist(),
                "dates_target": future_dates,
                "values_target": target.tolist(),
                "metrics": met,
            }
        else:
            _log("  run_sim start")
            res = run_sim(p, SimConfig(disease=args.disease, train_until=train_until, end=end_date, horizon=horizon, season_profile="flu"))
        # skip steps with empty target to avoid NaN metrics
        if not res.get("values_target"):
            _log("  skipped: empty target")
            continue
        last_params = p
        last_metrics = res.get("metrics", {})
        try:
            recent_metrics_window.append(last_metrics)
            if len(recent_metrics_window) > 8:
                recent_metrics_window.pop(0)
        except Exception:
            pass
        try:
            m = last_metrics or {}
            _log(f"  metrics: mae_median={m.get('mae_median')}, smape={m.get('smape')}, coverage95={m.get('coverage95')}, crps={m.get('crps')}")
        except Exception:
            pass
        steps.append({
            "train_until": train_until,
            "horizon": res.get("horizon"),
            "quantiles": res.get("quantiles"),
            "future_dates": res.get("future_dates"),
            "dates_hist": res.get("dates_hist"),
            "values_hist": res.get("values_hist"),
            "dates_target": res.get("dates_target"),
            "values_target": res.get("values_target"),
            "metrics": res.get("metrics"),
            "params": p,
            "llm_rationale": llm_rationale,
            "llm_proposal": llm_proposal,
        })
    # aggregate 1-week-ahead predictions per train_until (use q50/q05/q95 at h=1)
    agg_llm = {"pred_dates": [], "q50": [], "q05": [], "q95": []}
    for s in steps:
        if s.get("quantiles", {}).get("q50"):
            q50 = s["quantiles"]["q50"][0] if len(s["quantiles"]["q50"])>0 else None
            q05 = s["quantiles"]["q05"][0] if len(s["quantiles"]["q05"])>0 else None
            q95 = s["quantiles"]["q95"][0] if len(s["quantiles"]["q95"])>0 else None
            if q50 is not None:
                agg_llm["pred_dates"].append(s["future_dates"][0])
                agg_llm["q50"].append(q50)
                agg_llm["q05"].append(q05)
                agg_llm["q95"].append(q95)

    # baseline: 단순 지속성(직전 주 값) 1주ahead
    agg_base = {"pred_dates": [], "q50": [], "q05": [], "q95": []}
    # reconstruct y from any step's hist+target (use latest)
    if steps:
        all_dates = steps[-1]["dates_hist"] + steps[-1]["dates_target"]
        all_values = steps[-1]["values_hist"] + steps[-1]["values_target"]
        for i in range(1, len(all_values)):
            agg_base["pred_dates"].append(all_dates[i])
            prev = all_values[i-1]
            agg_base["q50"].append(prev)
            agg_base["q05"].append(prev)
            agg_base["q95"].append(prev)
    out_file = out_dir / f"rolling_{args.disease}_{args.start}_{args.end}_{args.n_steps}x{args.horizon}_{run_id}.html"
    out_file.write_text(build_html(args.disease, steps, agg_llm, agg_base), encoding="utf-8")
    print(str(out_file))
    # optional JSON dump
    if args.save_json:
        json_file = out_dir / f"rolling_{args.disease}_{args.start}_{args.end}_{args.n_steps}x{args.horizon}_{run_id}.json"
        import json as _json
        payload = {"disease": args.disease, "steps": steps, "agg_llm": agg_llm, "agg_base": agg_base}
        json_file.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(json_file))
    _log("done")


if __name__ == "__main__":
    main()


