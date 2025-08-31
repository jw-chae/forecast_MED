from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from Tools.adapters import load_his_outpatient_series
from Tools.llm_agent import propose_params_via_llm_with_debug, clamp_params, apply_hard_guards
from Tools.run_sim_wrapper import run_sim, SimConfig
from Tools.evidence_pack import (
    build_evidence_pack_with_web,
    build_evidence_pack_with_weather,
    build_evidence_pack_from_gov_monthly_file
)
from Tools.state_builder import build_observation


@dataclass
class RewardWeights:
    w1: float = 1.0   # CRPS
    w2: float = 50.0  # coverage shortfall
    w3: float = 0.2   # MAE_median
    w4: float = 10.0  # Peak recall penalty
    w5: float = 20.0  # KPI exceed bias


def iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def compute_reward(metrics: Dict[str, Any], target_cov: tuple[float, float] = (0.90, 0.98), target_recall: float = 0.8, weights: RewardWeights = RewardWeights()) -> float:
    crps = float(metrics.get("crps", 0.0))
    mae = float(metrics.get("mae_median", 0.0))
    cov = float(metrics.get("coverage95", 0.0))
    recall = float(metrics.get("recall_pm2w", 0.0))
    p_bed = float(metrics.get("p_bed_gt_0_92", 0.0))

    cov_short = max(0.0, target_cov[0] - cov)
    peak_pen = max(0.0, target_recall - recall) * 1.0
    kpi_bias = max(0.0, p_bed - 0.50) * 1.0

    loss = (
        weights.w1 * crps
        + weights.w2 * cov_short
        + weights.w3 * mae
        + weights.w4 * peak_pen
        + weights.w5 * kpi_bias
    )
    return float(-loss)


def weekly_dates_between(dates: List[str], start: str, end: str) -> List[str]:
    return [d for d in dates if start <= d <= end]


def build_offline_dataset(
    disease: str,
    start: str,
    end: str,
    horizon: int = 1,
    use_web: bool = True,
    gov_monthly_file: Optional[str] = None,
    regions: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> str:
    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    dt_index, series = load_his_outpatient_series(str(csv_path), disease)
    dates = [d.strftime("%Y-%m-%d") for d in dt_index]

    steps = weekly_dates_between(dates, start, end)
    out_root = Path(out_dir or (base / "reports" / "offline_dataset"))
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / f"ppo_dataset_{disease}_{start}_{end}_H{horizon}.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    # Use the same initial state as rolling_agent_forecast
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


    for i, train_until in enumerate(steps):
        end_dt = datetime.strptime(train_until, "%Y-%m-%d") + timedelta(weeks=horizon)
        step_end = min(datetime.strptime(end, "%Y-%m-%d"), end_dt)
        step_end_iso = iso(step_end)

        # evidence (web + weather + gov_file)
        ev = None
        if gov_monthly_file:
            try:
                ev = build_evidence_pack_from_gov_monthly_file(
                    gov_monthly_file,
                    base={"context_meta": {"disease": disease}},
                    asof=train_until,
                    future_weeks=horizon,
                )
            except Exception as e:
                print(f"[WARN] Failed to process gov_monthly_file: {e}")
        elif use_web:
            ev = build_evidence_pack_with_web(
                disease,
                regions or ["中国 全国", "浙江省"],
                base={},
                asof=train_until,
            )
        ev = build_evidence_pack_with_weather(base=(ev or {}), asof=train_until, location="hangzhou")


        # observation for LLM using the centralized builder
        observation = build_observation(
            disease=disease,
            train_until=train_until,
            end_date=step_end_iso,
            last_params=last_params,
            last_metrics=last_metrics,
            constraints=constraints,
            evidence=ev,
        )

        dbg = propose_params_via_llm_with_debug(
            observation=observation, provider="dashscope", model="qwen-max"
        ) or {}
        proposal = (dbg.get("proposal") if isinstance(dbg, dict) else None) or {}

        # simulation
        params = dict(last_params)
        params.update({k: v for k, v in proposal.items() if v is not None})
        params = apply_hard_guards(params)
        params = clamp_params(params, constraints)

        res = run_sim(params, SimConfig(disease=disease, train_until=train_until, end=step_end_iso, season_profile="flu", horizon=horizon))
        metrics = res.get("metrics", {})
        reward = compute_reward(metrics)

        record = {
            "iter": i,
            "timestamp": iso(datetime.utcnow()),
            "disease": disease,
            "period": f"{train_until}→{iso(step_end)}",
            "observation": observation,
            "action": proposal,
            "metrics": metrics,
            "reward": reward,
            "llm_raw": dbg.get("raw") if isinstance(dbg, dict) else None,
            "llm_usage": (dbg.get("response") or {}).get("usage") if isinstance(dbg, dict) else None,
        }
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        last_params = params
        last_metrics = metrics

    return str(jsonl_path)


def build_and_render(disease: str, start: str, end: str, horizon: int = 1, use_web: bool = True, gov_monthly_file: Optional[str] = None) -> Dict[str, str]:
    path = build_offline_dataset(disease, start, end, horizon=horizon, use_web=use_web, gov_monthly_file=gov_monthly_file)
    # simple HTML render
    data = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    xs = [r["period"].split("→")[0] for r in data]
    cov = [r["metrics"].get("coverage95") for r in data]
    crps = [r["metrics"].get("crps") for r in data]
    rew = [r.get("reward") for r in data]
    plotly = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    html = f"""
<!DOCTYPE html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"/><title>PPO Dataset · {disease}</title>
<script src=\"{plotly}\"></script></head><body>
<h2>PPO Offline Dataset · {disease}</h2>
<div id=\"cov\" style=\"height:320px\"></div>
<div id=\"crps\" style=\"height:320px\"></div>
<div id=\"rew\" style=\"height:320px\"></div>
<script>
var xs = {json.dumps(xs, ensure_ascii=False)};
var cov = {json.dumps(cov)};
var crps = {json.dumps(crps)};
var rew = {json.dumps(rew)};
Plotly.newPlot('cov', [{{x: xs, y: cov, mode:'lines+markers', name:'coverage95'}}], {{title:'coverage95'}}, {{displaylogo:false}});
Plotly.newPlot('crps', [{{x: xs, y: crps, mode:'lines+markers', name:'CRPS'}}], {{title:'CRPS'}}, {{displaylogo:false}});
Plotly.newPlot('rew', [{{x: xs, y: rew, mode:'lines+markers', name:'reward'}}], {{title:'reward'}}, {{displaylogo:false}});
</script>
</body></html>
"""
    out_html = Path(path).with_suffix(".html")
    out_html.write_text(html, encoding="utf-8")
    return {"jsonl": path, "html": str(out_html)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="手足口病")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--no_web", action="store_true")
    ap.add_argument("--gov_monthly_file", type=str, default=None, help="Path to government monthly stats file (CSV or JSONL).")
    args = ap.parse_args()
    res = build_and_render(
        args.disease,
        args.start,
        args.end,
        horizon=args.horizon,
        use_web=(not args.no_web),
        gov_monthly_file=args.gov_monthly_file
    )
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()


